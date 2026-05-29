"""PostgreSQL statistics reader and writer."""

from typing import List, Optional, Tuple

import pandas as pd

from numbers_dont_lie.models import ColumnStats, Constraint, ForeignKey, TableStats
from numbers_dont_lie.readers import BaseStatsReader, BaseWriter


class PostgresStatsReader(BaseStatsReader):
    """Read table statistics from PostgreSQL using system catalog views."""

    def __init__(self, host: str, port: int, database: str, user: str, password: str):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self._conn = None

    def connect(self) -> None:
        import psycopg2

        self._conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            dbname=self.database,
            user=self.user,
            password=self.password,
        )

    def disconnect(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def list_tables(self, schema: Optional[str] = None) -> List[Tuple[str, str, str]]:
        query = """
            SELECT table_catalog, table_schema, table_name
            FROM information_schema.tables
            WHERE table_type = 'BASE TABLE'
              AND table_schema NOT IN ('pg_catalog', 'information_schema')
        """
        params = []
        if schema:
            query += " AND table_schema = %s"
            params.append(schema)
        query += " ORDER BY table_schema, table_name"

        with self._conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()

    def read_table_stats(self, schema: str, table: str) -> TableStats:
        # Get row count estimate from pg_class (no full table scan)
        row_count = self._get_row_count(schema, table)

        # Get column info and stats
        columns = self._get_column_stats(schema, table, row_count)

        # Get FK and constraint metadata
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
        query = """
            SELECT COALESCE(c.reltuples::bigint, 0) AS row_estimate
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = %s AND c.relname = %s
        """
        with self._conn.cursor() as cur:
            cur.execute(query, (schema, table))
            row = cur.fetchone()
            return max(row[0], 0) if row else 0

    def _get_column_stats(self, schema: str, table: str, row_count: int) -> List[ColumnStats]:
        # Get column definitions
        col_query = """
            SELECT column_name, data_type, is_nullable, character_maximum_length
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
        """
        # Get pg_stats for distribution info
        stats_query = """
            SELECT attname, n_distinct, null_frac, avg_width
            FROM pg_stats
            WHERE schemaname = %s AND tablename = %s
        """
        with self._conn.cursor() as cur:
            cur.execute(col_query, (schema, table))
            col_rows = cur.fetchall()

            cur.execute(stats_query, (schema, table))
            stats_rows = {row[0]: row for row in cur.fetchall()}

        columns = []
        for col_name, data_type, is_nullable, max_length in col_rows:
            stat = stats_rows.get(col_name)

            distinct_count = None
            null_count = 0
            avg_length = None

            if stat:
                _, n_distinct, null_frac, avg_width = stat
                null_count = int(null_frac * row_count) if null_frac else 0
                avg_length = avg_width

                # n_distinct: positive = exact count, negative = fraction of rows
                if n_distinct and n_distinct > 0:
                    distinct_count = int(n_distinct)
                elif n_distinct and n_distinct < 0:
                    distinct_count = int(abs(n_distinct) * row_count)

            columns.append(ColumnStats(
                name=col_name,
                data_type=data_type,
                nullable=(is_nullable == "YES"),
                distinct_count=distinct_count,
                null_count=null_count,
                avg_length=avg_length,
                max_length=max_length,
            ))

        # Get min/max from pg_stats histogram bounds where available
        self._enrich_min_max(schema, table, columns)

        return columns

    def _enrich_min_max(self, schema: str, table: str, columns: List[ColumnStats]) -> None:
        """Get min/max estimates from histogram bounds in pg_stats."""
        query = """
            SELECT attname,
                   (SELECT min(val) FROM unnest(histogram_bounds::text::text[]) val) as min_val,
                   (SELECT max(val) FROM unnest(histogram_bounds::text::text[]) val) as max_val
            FROM pg_stats
            WHERE schemaname = %s AND tablename = %s
              AND histogram_bounds IS NOT NULL
        """
        try:
            with self._conn.cursor() as cur:
                cur.execute(query, (schema, table))
                for attname, min_val, max_val in cur.fetchall():
                    for col in columns:
                        if col.name == attname:
                            col.min_value = min_val
                            col.max_value = max_val
                            break
        except Exception:
            pass  # histogram bounds not always available

    def _get_foreign_keys(self, schema: str, table: str) -> List[ForeignKey]:
        """Get foreign key relationships from pg_constraint."""
        query = """
            SELECT
                con.conname AS constraint_name,
                child_ns.nspname AS child_schema,
                child_rel.relname AS child_table,
                ARRAY(
                    SELECT attname FROM pg_attribute
                    WHERE attrelid = con.conrelid AND attnum = ANY(con.conkey)
                ) AS child_columns,
                parent_ns.nspname AS parent_schema,
                parent_rel.relname AS parent_table,
                ARRAY(
                    SELECT attname FROM pg_attribute
                    WHERE attrelid = con.confrelid AND attnum = ANY(con.confkey)
                ) AS parent_columns
            FROM pg_constraint con
            JOIN pg_class child_rel ON con.conrelid = child_rel.oid
            JOIN pg_namespace child_ns ON child_rel.relnamespace = child_ns.oid
            JOIN pg_class parent_rel ON con.confrelid = parent_rel.oid
            JOIN pg_namespace parent_ns ON parent_rel.relnamespace = parent_ns.oid
            WHERE con.contype = 'f'
              AND child_ns.nspname = %s
              AND child_rel.relname = %s
        """
        foreign_keys = []
        with self._conn.cursor() as cur:
            cur.execute(query, (schema, table))
            for row in cur.fetchall():
                foreign_keys.append(ForeignKey(
                    name=row[0],
                    child_schema=row[1],
                    child_table=row[2],
                    child_columns=row[3],
                    parent_schema=row[4],
                    parent_table=row[5],
                    parent_columns=row[6],
                ))
        return foreign_keys

    def _get_constraints(self, schema: str, table: str) -> List[Constraint]:
        """Get PRIMARY KEY and UNIQUE constraints from pg_constraint."""
        query = """
            SELECT
                con.conname AS constraint_name,
                CASE con.contype
                    WHEN 'p' THEN 'PRIMARY KEY'
                    WHEN 'u' THEN 'UNIQUE'
                    WHEN 'c' THEN 'CHECK'
                END AS constraint_type,
                ARRAY(
                    SELECT attname FROM pg_attribute
                    WHERE attrelid = con.conrelid AND attnum = ANY(con.conkey)
                ) AS columns,
                pg_get_constraintdef(con.oid) AS check_expr
            FROM pg_constraint con
            JOIN pg_class rel ON con.conrelid = rel.oid
            JOIN pg_namespace ns ON rel.relnamespace = ns.oid
            WHERE con.contype IN ('p', 'u', 'c')
              AND ns.nspname = %s
              AND rel.relname = %s
        """
        constraints = []
        with self._conn.cursor() as cur:
            cur.execute(query, (schema, table))
            for row in cur.fetchall():
                constraints.append(Constraint(
                    name=row[0],
                    constraint_type=row[1],
                    columns=row[2],
                    check_expression=row[3] if row[1] == "CHECK" else None,
                ))
        return constraints


class PostgresWriter(BaseWriter):
    """Write synthetic data to PostgreSQL."""

    def __init__(self, host: str, port: int, database: str, user: str, password: str):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self._conn = None

    def connect(self) -> None:
        import psycopg2

        self._conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            dbname=self.database,
            user=self.user,
            password=self.password,
        )

    def disconnect(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def write_table(self, df: pd.DataFrame, schema: str, table: str, if_exists: str = "replace") -> int:
        from sqlalchemy import create_engine

        url = f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        engine = create_engine(url)

        df.to_sql(table, engine, schema=schema, if_exists=if_exists, index=False)
        engine.dispose()
        return len(df)
