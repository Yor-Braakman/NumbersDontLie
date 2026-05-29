from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class ColumnStats:
    """Statistics for a single column."""

    name: str
    data_type: str
    nullable: bool
    distinct_count: Optional[int] = None
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    null_count: int = 0
    avg_length: Optional[float] = None
    max_length: Optional[int] = None
    is_unique: bool = False

    @property
    def null_ratio(self) -> float:
        if self.distinct_count is None:
            return 0.0
        total = self.distinct_count + self.null_count
        return self.null_count / total if total > 0 else 0.0

    @property
    def is_categorical(self) -> bool:
        if self.data_type not in ("string", "varchar", "char", "text"):
            return False
        if self.distinct_count is None:
            return False
        total_count = self.distinct_count + self.null_count
        return total_count > 0 and (self.distinct_count / total_count) < 0.05


@dataclass
class ForeignKey:
    """A foreign key relationship between tables."""

    # Child (referencing) side
    child_schema: str
    child_table: str
    child_columns: List[str]

    # Parent (referenced) side
    parent_schema: str
    parent_table: str
    parent_columns: List[str]

    # Relationship name (constraint name)
    name: Optional[str] = None

    @property
    def child_full_name(self) -> str:
        return f"{self.child_schema}.{self.child_table}"

    @property
    def parent_full_name(self) -> str:
        return f"{self.parent_schema}.{self.parent_table}"


@dataclass
class Constraint:
    """A table constraint (unique, check, primary key)."""

    name: str
    constraint_type: str  # "PRIMARY KEY", "UNIQUE", "CHECK"
    columns: List[str]
    check_expression: Optional[str] = None  # For CHECK constraints


@dataclass
class TableStats:
    """Complete statistics for a table."""

    database_name: str
    schema_name: str
    table_name: str
    num_records: int
    columns: List[ColumnStats] = field(default_factory=list)
    foreign_keys: List[ForeignKey] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)

    @property
    def full_name(self) -> str:
        parts = [p for p in [self.database_name, self.schema_name, self.table_name] if p]
        return ".".join(parts)

    def get_column(self, name: str) -> Optional[ColumnStats]:
        for col in self.columns:
            if col.name == name:
                return col
        return None

    def get_unique_columns(self) -> List[List[str]]:
        """Get all sets of columns that must be unique (PK + UNIQUE constraints + is_unique columns)."""
        unique_sets = []
        for constraint in self.constraints:
            if constraint.constraint_type in ("PRIMARY KEY", "UNIQUE"):
                unique_sets.append(constraint.columns)
        # Also include individual columns marked as unique
        for col in self.columns:
            if col.is_unique and [col.name] not in unique_sets:
                unique_sets.append([col.name])
        return unique_sets

    def get_fk_for_column(self, column_name: str) -> Optional[ForeignKey]:
        """Get foreign key that references this column (on child side)."""
        for fk in self.foreign_keys:
            if column_name in fk.child_columns:
                return fk
        return None
