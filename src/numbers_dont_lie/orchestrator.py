"""Orchestrator for the synthetic data generation pipeline."""

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from numbers_dont_lie.generator import SyntheticDataGenerator
from numbers_dont_lie.models import TableStats
from numbers_dont_lie.readers import BaseStatsReader, BaseWriter


@dataclass
class TableResult:
    """Result of processing a single table."""

    table: str
    status: str  # "SUCCESS" or "FAILED"
    source_rows: int = 0
    generated_rows: int = 0
    error: Optional[str] = None


class Orchestrator:
    """Orchestrates the end-to-end synthetic data generation pipeline."""

    def __init__(
        self,
        reader: BaseStatsReader,
        writer: BaseWriter,
        scale_factor: float = 1.0,
        seed: int | None = None,
    ):
        self.reader = reader
        self.writer = writer
        self.generator = SyntheticDataGenerator(scale_factor=scale_factor, seed=seed)
        self.scale_factor = scale_factor

    def run(
        self,
        schema: Optional[str] = None,
        tables: Optional[List[str]] = None,
        dest_schema: str = "synthetic",
        progress_callback=None,
    ) -> List[TableResult]:
        """
        Execute the full pipeline.

        Args:
            schema: Source schema to read from (None = all schemas).
            tables: Specific table names to process (None = all in schema).
            dest_schema: Destination schema name for synthetic tables.
            progress_callback: Optional callable(current, total, table_name) for progress updates.
        """
        results = []

        # Discover tables
        all_tables = self.reader.list_tables(schema)

        if tables:
            all_tables = [(db, s, t) for db, s, t in all_tables if t in tables]

        total = len(all_tables)
        if total == 0:
            return results

        for i, (db, src_schema, table_name) in enumerate(all_tables):
            if progress_callback:
                progress_callback(i, total, table_name)

            try:
                # Read statistics
                table_stats = self.reader.read_table_stats(src_schema, table_name)

                # Generate synthetic data
                synthetic_df = self.generator.generate(table_stats)

                # Write to destination
                rows_written = self.writer.write_table(synthetic_df, dest_schema, table_name)

                results.append(TableResult(
                    table=f"{src_schema}.{table_name}",
                    status="SUCCESS",
                    source_rows=table_stats.num_records,
                    generated_rows=rows_written,
                ))

            except Exception as e:
                results.append(TableResult(
                    table=f"{src_schema}.{table_name}",
                    status="FAILED",
                    error=str(e),
                ))

        if progress_callback:
            progress_callback(total, total, "Done")

        return results
