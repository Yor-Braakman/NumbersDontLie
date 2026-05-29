"""Abstract base classes for database readers and writers."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from numbers_dont_lie.models import TableStats

import pandas as pd


class BaseStatsReader(ABC):
    """Abstract base for reading table statistics from any database."""

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the database."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the database."""

    @abstractmethod
    def list_tables(self, schema: Optional[str] = None) -> List[Tuple[str, str, str]]:
        """
        List available tables.

        Returns list of (database, schema, table) tuples.
        """

    @abstractmethod
    def read_table_stats(self, schema: str, table: str) -> TableStats:
        """
        Read statistics for a single table without loading data rows.

        Privacy guarantee: implementations must NOT read actual data rows.
        """

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


class BaseWriter(ABC):
    """Abstract base for writing synthetic data to a database."""

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the database."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the database."""

    @abstractmethod
    def write_table(self, df: pd.DataFrame, schema: str, table: str, if_exists: str = "replace") -> int:
        """
        Write a DataFrame to a database table.

        Returns the number of rows written.
        """

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
