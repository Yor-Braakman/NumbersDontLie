"""Generic ODBC statistics reader and writer using pyodbc.

Supports any database with an ODBC driver and information_schema support:
SQL Server, Azure SQL, Snowflake, Redshift, MariaDB, Vertica, CockroachDB, etc.
"""

from typing import List, Optional, Tuple

import pandas as pd

from numbers_dont_lie.models import ColumnStats, Constraint, ForeignKey, TableStats
from numbers_dont_lie.readers import BaseStatsReader, BaseWriter


class ODBCStatsReader(BaseStatsReader):
    """Read table statistics via ODBC using information_schema (SQL standard).

    Works with any database that supports information_schema views:
    SQL Server, Azure SQL, Snowflake, Redshift, MariaDB, Vertica, CockroachDB, etc.
    """

    def __init__(self, connection_string: str, database: Optional[str] = None):
        """
        Args:
            connection_string: Full ODBC connection string, e.g.:
                "DRIVER={ODBC Driver 18 for SQL Server};SERVER=localhost;DATABASE=mydb;UID=sa;PWD=pass"
                "DRIVER={Snowflake};SERVER=account.snowflakecomputing.com;DATABASE=mydb;UID=user;PWD=pass"
            database: Database name (used in metadata queries). Auto-detected if omitted.
        """
        self.connection_string = connection_string
        self.database = database
        self._conn = None

    def connect(self) -> None:
        import pyodbc

        self._conn = pyodbc.connect(self.connection_string)
        if not self.database:
            # Try to auto-detect current database
            try:
                cur = self._conn.cursor()
                cur.execute("SELECT DB_NAME()")  # SQL Server
                self.database = cur.fetchone()[0]
                cur.close()
            except Exception:
                try:
                    cur = self._conn.cursor()
                    cur.execute("SELECT CURRENT_DATABASE()")  # Snowflake/PG/Redshift
                    self.database = cur.fetchone()[0]
                    cur.close()
                except Exception:
                    self.database = ""

    def disconnect(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def list_tables(self, schema: Optional[str] = None) -> List[Tuple[str, str, str]]:
        query = """
            SELECT TABLE_CATALOG, TABLE_SCHEMA, TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_TYPE = 'BASE TABLE'
        """
        params = []
        if schema:
            query += " AND TABLE_SCHEMA = ?"
            params.append(schema)

        # Exclude system schemas common across databases
        query += """
            AND TABLE_SCHEMA NOT IN (
                'INFORMATION_SCHEMA', 'information_schema',
                'pg_catalog', 'sys', 'INFORMATION_SCHEMA'
            )
            ORDER BY TABLE_SCHEMA, TABLE_NAME
        """

        cur = self._conn.cursor()
        cur.execute(query, params)
        results = [(row[0] or self.database, row[1], row[2]) for row in cur.fetchall()]
        cur.close()
        return results

    def read_table_stats(self, schema: str, table: str) -> TableStats:
        row_count = self._get_row_count(schema, table)
        columns = self._get_column_stats(schema, table, row_count)
        foreign_keys = self._get_foreign_keys(schema, table)
        constraints = self._get_constraints(schema, table)

        # Mark unique columns
        for constraint in constraints:
            if constraint.constraint_type in ("PRIMARY KEY", "UNIQUE"):
                if len(constraint.columns) == 1:
                    col = next((c for c in columns if c.name == constraint.columns[0]), None)
                    if col:
                        col.is_unique = True

        return TableStats(
            database_name=self.database or "",
            schema_name=schema,
            table_name=table,
            num_records=row_count,
            columns=columns,
            foreign_keys=foreign_keys,
            constraints=constraints,
        )

    def _get_row_count(self, schema: str, table: str) -> int:
        """Get row count - try system views first, fall back to COUNT."""
        # Try SQL Server sys.dm_db_partition_stats (fast, no table scan)
        try:
            query = """
                SELECT SUM(p.rows)
                FROM sys.partitions p
                JOIN sys.tables t ON p.object_id = t.object_id
                JOIN sys.schemas s ON t.schema_id = s.schema_id
                WHERE s.name = ? AND t.name = ? AND p.index_id IN (0, 1)
            """
            cur = self._conn.cursor()
            cur.execute(query, (schema, table))
            row = cur.fetchone()
            cur.close()
            if row and row[0]:
                return int(row[0])
        except Exception:
            pass

        # Fall back to COUNT (works everywhere)
        try:
            # Use quoted identifiers for safety
            quoted = self._quote_identifier(schema, table)
            cur = self._conn.cursor()
            cur.execute(f"SELECT COUNT(*) FROM {quoted}")
            count = cur.fetchone()[0]
            cur.close()
            return count
        except Exception:
            return 0

    def _get_column_stats(self, schema: str, table: str, row_count: int) -> List[ColumnStats]:
        # Get column definitions from information_schema
        col_query = """
            SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, CHARACTER_MAXIMUM_LENGTH
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
            ORDER BY ORDINAL_POSITION
        """
        cur = self._conn.cursor()
        cur.execute(col_query, (schema, table))
        col_rows = cur.fetchall()
        cur.close()

        columns = []
        for col_name, data_type, is_nullable, max_length in col_rows:
            # Get per-column stats via aggregate query
            stats = self._get_single_column_stats(schema, table, col_name)

            columns.append(ColumnStats(
                name=col_name,
                data_type=data_type.lower() if data_type else "varchar",
                nullable=(is_nullable in ("YES", "yes", True)),
                distinct_count=stats.get("distinct_count"),
                min_value=stats.get("min_value"),
                max_value=stats.get("max_value"),
                null_count=stats.get("null_count", 0),
                avg_length=stats.get("avg_length"),
                max_length=max_length,
            ))

        # Try to enrich with database-specific stats (SQL Server)
        self._enrich_with_sys_stats(schema, table, columns)

        return columns

    def _get_single_column_stats(self, schema: str, table: str, column: str) -> dict:
        """Get basic stats for a column using aggregate queries."""
        quoted_table = self._quote_identifier(schema, table)
        quoted_col = self._quote_column(column)

        query = f"""
            SELECT
                COUNT(DISTINCT {quoted_col}) AS distinct_count,
                MIN({quoted_col}) AS min_val,
                MAX({quoted_col}) AS max_val,
                SUM(CASE WHEN {quoted_col} IS NULL THEN 1 ELSE 0 END) AS null_count,
                AVG(CAST(LEN({quoted_col}) AS FLOAT)) AS avg_length
            FROM {quoted_table}
        """
        # Some databases use LENGTH instead of LEN
        try:
            cur = self._conn.cursor()
            cur.execute(query)
            row = cur.fetchone()
            cur.close()
            return {
                "distinct_count": row[0],
                "min_value": row[1],
                "max_value": row[2],
                "null_count": row[3] or 0,
                "avg_length": row[4],
            }
        except Exception:
            # Retry with LENGTH (PostgreSQL/Snowflake/Redshift)
            query_alt = f"""
                SELECT
                    COUNT(DISTINCT {quoted_col}),
                    MIN({quoted_col}),
                    MAX({quoted_col}),
                    SUM(CASE WHEN {quoted_col} IS NULL THEN 1 ELSE 0 END),
                    AVG(LENGTH(CAST({quoted_col} AS VARCHAR)))
                FROM {quoted_table}
            """
            try:
                cur = self._conn.cursor()
                cur.execute(query_alt)
                row = cur.fetchone()
                cur.close()
                return {
                    "distinct_count": row[0],
                    "min_value": row[1],
                    "max_value": row[2],
                    "null_count": row[3] or 0,
                    "avg_length": row[4],
                }
            except Exception:
                return {}

    def _enrich_with_sys_stats(self, schema: str, table: str, columns: List[ColumnStats]) -> None:
        """Try to get richer stats from SQL Server system views."""
        try:
            query = """
                SELECT
                    c.name AS column_name,
                    sp.rows AS row_count,
                    sp.rows_sampled,
                    sp.modification_counter
                FROM sys.stats s
                CROSS APPLY sys.dm_db_stats_properties(s.object_id, s.stats_id) sp
                JOIN sys.stats_columns sc ON s.object_id = sc.object_id AND s.stats_id = sc.stats_id
                JOIN sys.columns c ON sc.object_id = c.object_id AND sc.column_id = c.column_id
                JOIN sys.tables t ON s.object_id = t.object_id
                JOIN sys.schemas sch ON t.schema_id = sch.schema_id
                WHERE sch.name = ? AND t.name = ?
            """
            cur = self._conn.cursor()
            cur.execute(query, (schema, table))
            cur.fetchall()  # consume results even if we don't use them all
            cur.close()
        except Exception:
            pass  # Not SQL Server or no permissions

    def _get_foreign_keys(self, schema: str, table: str) -> List[ForeignKey]:
        """Get foreign key relationships via information_schema."""
        query = """
            SELECT
                tc.CONSTRAINT_NAME,
                kcu.COLUMN_NAME,
                ccu.TABLE_SCHEMA AS REFERENCED_SCHEMA,
                ccu.TABLE_NAME AS REFERENCED_TABLE,
                ccu.COLUMN_NAME AS REFERENCED_COLUMN
            FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
            JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
              ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
              AND tc.TABLE_SCHEMA = kcu.TABLE_SCHEMA
            JOIN INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS rc
              ON tc.CONSTRAINT_NAME = rc.CONSTRAINT_NAME
            JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE ccu
              ON rc.UNIQUE_CONSTRAINT_NAME = ccu.CONSTRAINT_NAME
              AND kcu.ORDINAL_POSITION = ccu.ORDINAL_POSITION
            WHERE tc.CONSTRAINT_TYPE = 'FOREIGN KEY'
              AND tc.TABLE_SCHEMA = ?
              AND tc.TABLE_NAME = ?
            ORDER BY tc.CONSTRAINT_NAME, kcu.ORDINAL_POSITION
        """
        try:
            cur = self._conn.cursor()
            cur.execute(query, (schema, table))
            rows = cur.fetchall()
            cur.close()
        except Exception:
            return []

        # Group by constraint name
        fk_groups = {}
        for constraint_name, col_name, ref_schema, ref_table, ref_col in rows:
            if constraint_name not in fk_groups:
                fk_groups[constraint_name] = {
                    "ref_schema": ref_schema,
                    "ref_table": ref_table,
                    "child_columns": [],
                    "parent_columns": [],
                }
            fk_groups[constraint_name]["child_columns"].append(col_name)
            fk_groups[constraint_name]["parent_columns"].append(ref_col)

        return [
            ForeignKey(
                name=name,
                child_schema=schema,
                child_table=table,
                child_columns=data["child_columns"],
                parent_schema=data["ref_schema"],
                parent_table=data["ref_table"],
                parent_columns=data["parent_columns"],
            )
            for name, data in fk_groups.items()
        ]

    def _get_constraints(self, schema: str, table: str) -> List[Constraint]:
        """Get PK, UNIQUE, and CHECK constraints via information_schema + sys views."""
        constraints = []

        # PK and UNIQUE from information_schema
        query = """
            SELECT tc.CONSTRAINT_NAME, tc.CONSTRAINT_TYPE, kcu.COLUMN_NAME
            FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
            JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
              ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
              AND tc.TABLE_SCHEMA = kcu.TABLE_SCHEMA
            WHERE tc.TABLE_SCHEMA = ? AND tc.TABLE_NAME = ?
              AND tc.CONSTRAINT_TYPE IN ('PRIMARY KEY', 'UNIQUE')
            ORDER BY tc.CONSTRAINT_NAME, kcu.ORDINAL_POSITION
        """
        try:
            cur = self._conn.cursor()
            cur.execute(query, (schema, table))
            rows = cur.fetchall()
            cur.close()

            groups = {}
            for name, ctype, col in rows:
                if name not in groups:
                    groups[name] = {"type": ctype, "columns": []}
                groups[name]["columns"].append(col)

            for name, data in groups.items():
                constraints.append(Constraint(
                    name=name,
                    constraint_type=data["type"],
                    columns=data["columns"],
                ))
        except Exception:
            pass

        # CHECK constraints from sys.check_constraints (SQL Server specific)
        try:
            check_query = """
                SELECT cc.name, cc.definition
                FROM sys.check_constraints cc
                JOIN sys.tables t ON cc.parent_object_id = t.object_id
                JOIN sys.schemas s ON t.schema_id = s.schema_id
                WHERE s.name = ? AND t.name = ?
            """
            cur = self._conn.cursor()
            cur.execute(check_query, (schema, table))
            for name, definition in cur.fetchall():
                # Try to extract column names from the definition
                col_names = self._extract_columns_from_check(definition, schema, table)
                constraints.append(Constraint(
                    name=name,
                    constraint_type="CHECK",
                    columns=col_names,
                    check_expression=definition,
                ))
            cur.close()
        except Exception:
            pass

        return constraints

    def _extract_columns_from_check(self, definition: str, schema: str, table: str) -> List[str]:
        """Best-effort extraction of column names referenced in a CHECK expression."""
        # Get all column names for this table
        try:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?",
                (schema, table),
            )
            all_columns = [row[0] for row in cur.fetchall()]
            cur.close()
            # Return columns that appear in the definition
            return [c for c in all_columns if f"[{c}]" in definition or c in definition]
        except Exception:
            return []

    def _quote_identifier(self, schema: str, table: str) -> str:
        """Quote a schema.table identifier safely."""
        # Use bracket quoting (SQL Server/Access) or double-quote (standard SQL)
        # Brackets work on SQL Server; double quotes are ANSI standard
        safe_schema = schema.replace('"', '""')
        safe_table = table.replace('"', '""')
        return f'"{safe_schema}"."{safe_table}"'

    def _quote_column(self, column: str) -> str:
        """Quote a column identifier safely."""
        safe_col = column.replace('"', '""')
        return f'"{safe_col}"'


class ODBCWriter(BaseWriter):
    """Write synthetic data via ODBC using pandas to_sql with SQLAlchemy."""

    def __init__(self, connection_string: str):
        """
        Args:
            connection_string: Full ODBC connection string.
        """
        self.connection_string = connection_string
        self._conn = None

    def connect(self) -> None:
        import pyodbc

        self._conn = pyodbc.connect(self.connection_string)

    def disconnect(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def write_table(self, df: pd.DataFrame, schema: str, table: str, if_exists: str = "replace") -> int:
        """Write DataFrame using fast_executemany for bulk inserts."""
        import pyodbc

        cursor = self._conn.cursor()
        cursor.fast_executemany = True

        # Drop and recreate if replacing
        quoted = f'"{schema}"."{table}"'
        if if_exists == "replace":
            try:
                cursor.execute(f"DROP TABLE IF EXISTS {quoted}")
                self._conn.commit()
            except Exception:
                pass

            # Create table from DataFrame schema
            create_sql = self._build_create_table(schema, table, df)
            cursor.execute(create_sql)
            self._conn.commit()

        # Insert in batches
        cols = list(df.columns)
        placeholders = ", ".join(["?"] * len(cols))
        col_list = ", ".join(f'"{c}"' for c in cols)
        insert_sql = f'INSERT INTO {quoted} ({col_list}) VALUES ({placeholders})'

        batch_size = 1000
        rows_written = 0
        data = df.values.tolist()

        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            cursor.executemany(insert_sql, batch)
            rows_written += len(batch)

        self._conn.commit()
        cursor.close()
        return rows_written

    def _build_create_table(self, schema: str, table: str, df: pd.DataFrame) -> str:
        """Build a CREATE TABLE statement from DataFrame dtypes."""
        quoted = f'"{schema}"."{table}"'
        col_defs = []
        for col_name in df.columns:
            dtype = df[col_name].dtype
            sql_type = self._pandas_dtype_to_sql(dtype, col_name, df)
            safe_col = col_name.replace('"', '""')
            col_defs.append(f'  "{safe_col}" {sql_type}')

        cols_sql = ",\n".join(col_defs)
        return f"CREATE TABLE {quoted} (\n{cols_sql}\n)"

    def _pandas_dtype_to_sql(self, dtype, col_name: str, df: pd.DataFrame) -> str:
        """Map pandas dtype to SQL type."""
        dtype_str = str(dtype)
        if "int" in dtype_str:
            return "BIGINT"
        elif "float" in dtype_str:
            return "FLOAT"
        elif "bool" in dtype_str:
            return "BIT"
        elif "datetime" in dtype_str:
            return "DATETIME2"
        elif "object" in dtype_str:
            # Estimate max length from data
            try:
                max_len = df[col_name].astype(str).str.len().max()
                max_len = max(max_len, 50) if max_len else 255
                return f"NVARCHAR({min(max_len * 2, 4000)})"
            except Exception:
                return "NVARCHAR(4000)"
        else:
            return "NVARCHAR(4000)"
