"""SQLite statistics reader and writer."""

import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from numbers_dont_lie.models import ColumnStats, Constraint, ForeignKey, TableStats
from numbers_dont_lie.readers import BaseStatsReader, BaseWriter


class SQLiteStatsReader(BaseStatsReader):
    """Read table statistics from SQLite using pragma and sqlite_stat tables."""

    def __init__(self, database_path: str):
        self.database_path = database_path
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        if not Path(self.database_path).exists():
            raise FileNotFoundError(f"Database not found: {self.database_path}")
        self._conn = sqlite3.connect(self.database_path)
        self._conn.row_factory = sqlite3.Row

    def disconnect(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def list_tables(self, schema: Optional[str] = None) -> List[Tuple[str, str, str]]:
        cur = self._conn.cursor()
        cur.execute("""
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        db_name = Path(self.database_path).stem
        return [(db_name, "main", row[0]) for row in cur.fetchall()]

    def read_table_stats(self, schema: str, table: str) -> TableStats:
        row_count = self._get_row_count(table)
        columns = self._get_column_stats(table, row_count)
        foreign_keys = self._get_foreign_keys(table, schema)
        constraints = self._get_constraints(table)

        # Mark unique columns
        for constraint in constraints:
            if constraint.constraint_type in ("PRIMARY KEY", "UNIQUE"):
                if len(constraint.columns) == 1:
                    col = next((c for c in columns if c.name == constraint.columns[0]), None)
                    if col:
                        col.is_unique = True

        db_name = Path(self.database_path).stem
        return TableStats(
            database_name=db_name,
            schema_name=schema,
            table_name=table,
            num_records=row_count,
            columns=columns,
            foreign_keys=foreign_keys,
            constraints=constraints,
        )

    def _get_row_count(self, table: str) -> int:
        """Get row count - try stat tables first, fall back to COUNT."""
        cur = self._conn.cursor()

        # Try sqlite_stat1 (populated by ANALYZE)
        try:
            cur.execute(
                "SELECT stat FROM sqlite_stat1 WHERE tbl = ? AND idx IS NULL",
                (table,),
            )
            row = cur.fetchone()
            if row:
                # stat column format: "nrows ..."
                return int(str(row[0]).split()[0])
        except sqlite3.OperationalError:
            pass

        # Fall back to COUNT (SQLite is fast for this)
        cur.execute(f'SELECT COUNT(*) FROM "{table}"')  # noqa: S608
        return cur.fetchone()[0]

    def _get_column_stats(self, table: str, row_count: int) -> List[ColumnStats]:
        cur = self._conn.cursor()
        cur.execute(f'PRAGMA table_info("{table}")')
        pragma_rows = cur.fetchall()

        columns = []
        for row in pragma_rows:
            col_name = row[1]
            data_type = row[2] or "text"
            nullable = row[3] == 0  # notnull column: 0 means nullable

            # Get basic stats via aggregate queries on metadata
            stats = self._get_single_column_stats(table, col_name, row_count)

            columns.append(ColumnStats(
                name=col_name,
                data_type=data_type.lower(),
                nullable=nullable,
                distinct_count=stats.get("distinct_count"),
                min_value=stats.get("min_value"),
                max_value=stats.get("max_value"),
                null_count=stats.get("null_count", 0),
                avg_length=stats.get("avg_length"),
                max_length=stats.get("max_length"),
            ))

        return columns

    def _get_single_column_stats(self, table: str, column: str, row_count: int) -> dict:
        """Get stats for a single column using aggregate queries."""
        cur = self._conn.cursor()

        # Use a single aggregate query for efficiency
        query = f"""
            SELECT
                COUNT(DISTINCT "{column}") as distinct_count,
                MIN("{column}") as min_val,
                MAX("{column}") as max_val,
                SUM(CASE WHEN "{column}" IS NULL THEN 1 ELSE 0 END) as null_count,
                AVG(LENGTH("{column}")) as avg_length,
                MAX(LENGTH("{column}")) as max_length
            FROM "{table}"
        """  # noqa: S608
        cur.execute(query)
        row = cur.fetchone()

        return {
            "distinct_count": row[0],
            "min_value": row[1],
            "max_value": row[2],
            "null_count": row[3] or 0,
            "avg_length": row[4],
            "max_length": row[5],
        }

    def _get_foreign_keys(self, table: str, schema: str) -> List[ForeignKey]:
        """Get foreign key relationships for a table."""
        cur = self._conn.cursor()
        cur.execute(f'PRAGMA foreign_key_list("{table}")')
        rows = cur.fetchall()

        # Group by FK id (column index 0)
        fk_groups = {}
        for row in rows:
            fk_id = row[0]
            if fk_id not in fk_groups:
                fk_groups[fk_id] = {
                    "parent_table": row[2],
                    "child_columns": [],
                    "parent_columns": [],
                }
            fk_groups[fk_id]["child_columns"].append(row[3])
            fk_groups[fk_id]["parent_columns"].append(row[4])

        foreign_keys = []
        for fk_id, fk_data in fk_groups.items():
            foreign_keys.append(ForeignKey(
                child_schema=schema,
                child_table=table,
                child_columns=fk_data["child_columns"],
                parent_schema=schema,
                parent_table=fk_data["parent_table"],
                parent_columns=fk_data["parent_columns"],
                name=f"fk_{table}_{fk_id}",
            ))

        return foreign_keys

    def _get_constraints(self, table: str) -> List[Constraint]:
        """Get unique and primary key constraints for a table."""
        constraints = []
        cur = self._conn.cursor()

        # Get primary key columns
        cur.execute(f'PRAGMA table_info("{table}")')
        pk_columns = [row[1] for row in cur.fetchall() if row[5] > 0]  # pk column > 0
        if pk_columns:
            constraints.append(Constraint(
                name=f"pk_{table}",
                constraint_type="PRIMARY KEY",
                columns=pk_columns,
            ))

        # Get unique indexes
        cur.execute(f'PRAGMA index_list("{table}")')
        for idx_row in cur.fetchall():
            idx_name = idx_row[1]
            is_unique = idx_row[2]
            if is_unique:
                cur.execute(f'PRAGMA index_info("{idx_name}")')
                idx_columns = [col_row[2] for col_row in cur.fetchall()]
                if idx_columns and idx_columns != pk_columns:
                    constraints.append(Constraint(
                        name=idx_name,
                        constraint_type="UNIQUE",
                        columns=idx_columns,
                    ))

        return constraints


class SQLiteWriter(BaseWriter):
    """Write synthetic data to SQLite."""

    def __init__(self, database_path: str):
        self.database_path = database_path
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        self._conn = sqlite3.connect(self.database_path)

    def disconnect(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def write_table(self, df: pd.DataFrame, schema: str, table: str, if_exists: str = "replace") -> int:
        df.to_sql(table, self._conn, if_exists=if_exists, index=False)
        return len(df)
