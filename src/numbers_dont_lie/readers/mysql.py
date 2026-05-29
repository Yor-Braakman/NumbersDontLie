"""MySQL statistics reader and writer."""

from typing import List, Optional, Tuple

import pandas as pd

from numbers_dont_lie.models import ColumnStats, Constraint, ForeignKey, TableStats
from numbers_dont_lie.readers import BaseStatsReader, BaseWriter


class MySQLStatsReader(BaseStatsReader):
    """Read table statistics from MySQL using information_schema."""

    def __init__(self, host: str, port: int, database: str, user: str, password: str):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self._conn = None

    def connect(self) -> None:
        import mysql.connector

        self._conn = mysql.connector.connect(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password,
        )

    def disconnect(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def list_tables(self, schema: Optional[str] = None) -> List[Tuple[str, str, str]]:
        target_schema = schema or self.database
        query = """
            SELECT TABLE_SCHEMA, TABLE_SCHEMA, TABLE_NAME
            FROM information_schema.TABLES
            WHERE TABLE_SCHEMA = %s AND TABLE_TYPE = 'BASE TABLE'
            ORDER BY TABLE_NAME
        """
        cur = self._conn.cursor()
        cur.execute(query, (target_schema,))
        results = [(self.database, row[1], row[2]) for row in cur.fetchall()]
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
            database_name=self.database,
            schema_name=schema,
            table_name=table,
            num_records=row_count,
            columns=columns,
            foreign_keys=foreign_keys,
            constraints=constraints,
        )

    def _get_row_count(self, schema: str, table: str) -> int:
        """Get row count estimate from information_schema.TABLES."""
        query = """
            SELECT TABLE_ROWS
            FROM information_schema.TABLES
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
        """
        cur = self._conn.cursor()
        cur.execute(query, (schema, table))
        row = cur.fetchone()
        cur.close()
        return row[0] if row and row[0] else 0

    def _get_column_stats(self, schema: str, table: str, row_count: int) -> List[ColumnStats]:
        # Get column definitions
        col_query = """
            SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, CHARACTER_MAXIMUM_LENGTH
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
            ORDER BY ORDINAL_POSITION
        """
        # Get index cardinality for distinct count estimates
        idx_query = """
            SELECT COLUMN_NAME, CARDINALITY
            FROM information_schema.STATISTICS
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
        """
        cur = self._conn.cursor()
        cur.execute(col_query, (schema, table))
        col_rows = cur.fetchall()

        cur.execute(idx_query, (schema, table))
        idx_stats = {row[0]: row[1] for row in cur.fetchall()}
        cur.close()

        columns = []
        for col_name, data_type, is_nullable, max_length in col_rows:
            distinct_count = idx_stats.get(col_name)

            # Get min/max from column statistics if available
            min_val, max_val, null_count = self._get_column_min_max(schema, table, col_name)

            columns.append(ColumnStats(
                name=col_name,
                data_type=data_type.lower(),
                nullable=(is_nullable == "YES"),
                distinct_count=int(distinct_count) if distinct_count else None,
                min_value=min_val,
                max_value=max_val,
                null_count=null_count,
                max_length=max_length,
            ))

        return columns

    def _get_column_min_max(self, schema: str, table: str, column: str) -> tuple:
        """Get min, max, and null count for a column using aggregate query."""
        # Use parameterized identifier quoting for MySQL
        query = f"""
            SELECT MIN(`{column}`), MAX(`{column}`),
                   SUM(CASE WHEN `{column}` IS NULL THEN 1 ELSE 0 END)
            FROM `{schema}`.`{table}`
        """
        try:
            cur = self._conn.cursor()
            cur.execute(query)
            row = cur.fetchone()
            cur.close()
            return (row[0], row[1], row[2] or 0) if row else (None, None, 0)
        except Exception:
            return (None, None, 0)

    def _get_foreign_keys(self, schema: str, table: str) -> List[ForeignKey]:
        """Get foreign key relationships from information_schema."""
        query = """
            SELECT
                CONSTRAINT_NAME,
                COLUMN_NAME,
                REFERENCED_TABLE_SCHEMA,
                REFERENCED_TABLE_NAME,
                REFERENCED_COLUMN_NAME
            FROM information_schema.KEY_COLUMN_USAGE
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
              AND REFERENCED_TABLE_NAME IS NOT NULL
            ORDER BY CONSTRAINT_NAME, ORDINAL_POSITION
        """
        cur = self._conn.cursor()
        cur.execute(query, (schema, table))
        rows = cur.fetchall()
        cur.close()

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
        """Get PRIMARY KEY and UNIQUE constraints from information_schema."""
        query = """
            SELECT
                tc.CONSTRAINT_NAME,
                tc.CONSTRAINT_TYPE,
                GROUP_CONCAT(kcu.COLUMN_NAME ORDER BY kcu.ORDINAL_POSITION) AS columns
            FROM information_schema.TABLE_CONSTRAINTS tc
            JOIN information_schema.KEY_COLUMN_USAGE kcu
              ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
              AND tc.TABLE_SCHEMA = kcu.TABLE_SCHEMA
              AND tc.TABLE_NAME = kcu.TABLE_NAME
            WHERE tc.TABLE_SCHEMA = %s AND tc.TABLE_NAME = %s
              AND tc.CONSTRAINT_TYPE IN ('PRIMARY KEY', 'UNIQUE')
            GROUP BY tc.CONSTRAINT_NAME, tc.CONSTRAINT_TYPE
        """
        cur = self._conn.cursor()
        cur.execute(query, (schema, table))
        rows = cur.fetchall()
        cur.close()

        return [
            Constraint(
                name=row[0],
                constraint_type=row[1],
                columns=row[2].split(","),
            )
            for row in rows
        ]


class MySQLWriter(BaseWriter):
    """Write synthetic data to MySQL."""

    def __init__(self, host: str, port: int, database: str, user: str, password: str):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self._conn = None

    def connect(self) -> None:
        import mysql.connector

        self._conn = mysql.connector.connect(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password,
        )

    def disconnect(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def write_table(self, df: pd.DataFrame, schema: str, table: str, if_exists: str = "replace") -> int:
        from sqlalchemy import create_engine

        url = f"mysql+mysqlconnector://{self.user}:{self.password}@{self.host}:{self.port}/{schema}"
        engine = create_engine(url)

        df.to_sql(table, engine, schema=schema, if_exists=if_exists, index=False)
        engine.dispose()
        return len(df)
