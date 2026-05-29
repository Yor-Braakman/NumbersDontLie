"""Synthetic data generator using pandas/numpy (no Spark dependency)."""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from numbers_dont_lie.check_constraints import CheckConstraintParser, CheckRule
from numbers_dont_lie.models import ColumnStats, ForeignKey, TableStats


class SyntheticDataGenerator:
    """Generate synthetic data from table statistics using pandas."""

    def __init__(self, scale_factor: float = 1.0, seed: int | None = None):
        self.scale_factor = scale_factor
        self.rng = np.random.default_rng(seed)
        # Store generated parent tables for FK lookups
        self._generated_tables: Dict[str, pd.DataFrame] = {}
        self._check_parser = CheckConstraintParser()

    def register_generated_table(self, table_full_name: str, df: pd.DataFrame) -> None:
        """Register a generated table so FK columns can reference it."""
        self._generated_tables[table_full_name] = df

    def generate(self, table_stats: TableStats) -> pd.DataFrame:
        """Generate a synthetic DataFrame based on table statistics."""
        target_rows = int(table_stats.num_records * self.scale_factor)
        if target_rows == 0:
            target_rows = 100  # minimum fallback

        # Parse CHECK constraints into column-level rules
        check_rules = self._parse_check_constraints(table_stats)

        data = {}

        # First pass: generate FK columns that reference parent tables
        fk_columns = set()
        for fk in table_stats.foreign_keys:
            parent_df = self._generated_tables.get(fk.parent_full_name)
            if parent_df is not None:
                # For composite FKs, sample the same row indices from parent
                parent_valid = parent_df[fk.parent_columns].dropna()
                if len(parent_valid) > 0:
                    row_indices = self.rng.integers(0, len(parent_valid), size=target_rows)
                    for child_col, parent_col in zip(fk.child_columns, fk.parent_columns):
                        fk_columns.add(child_col)
                        values = parent_valid[parent_col].values[row_indices]
                        series = pd.Series(values)

                        # Apply null injection if nullable
                        col_stats = table_stats.get_column(child_col)
                        if col_stats and col_stats.nullable and col_stats.null_ratio > 0:
                            mask = self.rng.random(target_rows) < col_stats.null_ratio
                            series = series.where(~mask, other=None)

                        data[child_col] = series

        # Second pass: generate remaining columns (CHECK-constraint-aware)
        for col_stats in table_stats.columns:
            if col_stats.name in fk_columns:
                continue
            col_rules = check_rules.get(col_stats.name, [])
            if col_rules:
                data[col_stats.name] = self._generate_column_with_checks(
                    col_stats, target_rows, col_rules
                )
            else:
                data[col_stats.name] = self._generate_column(col_stats, target_rows)

        df = pd.DataFrame(data)

        # Third pass: enforce cross-column CHECK constraints
        cross_rules = check_rules.get("__cross_column__", [])
        if cross_rules:
            df = self._enforce_cross_column_checks(df, cross_rules, table_stats, target_rows)

        # Fourth pass: enforce unique constraints
        unique_sets = table_stats.get_unique_columns()
        for unique_cols in unique_sets:
            df = self._enforce_unique(df, unique_cols, table_stats)

        # Register this table for downstream FK references
        table_key = f"{table_stats.schema_name}.{table_stats.table_name}"
        self._generated_tables[table_key] = df

        return df

    def _parse_check_constraints(self, table_stats: TableStats) -> Dict[str, List[CheckRule]]:
        """Parse all CHECK constraints into per-column rule sets."""
        col_names = [c.name for c in table_stats.columns]
        rules_by_column: Dict[str, List[CheckRule]] = {}

        for constraint in table_stats.constraints:
            if constraint.constraint_type != "CHECK" or not constraint.check_expression:
                continue
            parsed = self._check_parser.parse(constraint.check_expression, col_names)
            for rule in parsed:
                if rule.rule_type == "cross_column":
                    rules_by_column.setdefault("__cross_column__", []).append(rule)
                else:
                    rules_by_column.setdefault(rule.column, []).append(rule)

        return rules_by_column

    def _generate_column_with_checks(
        self, col_stats: ColumnStats, num_rows: int, rules: List[CheckRule]
    ) -> pd.Series:
        """Generate column values that satisfy CHECK constraint rules."""
        # Priority: in_list > between > like_pattern > comparison > not_like (filter)
        in_list_rule = next((r for r in rules if r.rule_type == "in_list"), None)
        between_rule = next((r for r in rules if r.rule_type == "between"), None)
        like_rules = [r for r in rules if r.rule_type == "like_pattern"]
        not_like_rules = [r for r in rules if r.rule_type == "not_like"]
        comparison_rules = [r for r in rules if r.rule_type == "comparison"]

        if in_list_rule:
            return self._generate_from_in_list(col_stats, num_rows, in_list_rule)

        if between_rule:
            return self._generate_from_between(col_stats, num_rows, between_rule)

        if like_rules:
            return self._generate_from_like(col_stats, num_rows, like_rules[0])

        # For comparisons, adjust min/max then use normal generation
        if comparison_rules:
            return self._generate_with_comparisons(col_stats, num_rows, comparison_rules)

        # For not_like only, generate normally then filter
        values = self._generate_column(col_stats, num_rows)
        if not_like_rules:
            values = self._filter_not_like(values, not_like_rules)
        return values

    def _generate_from_in_list(
        self, col_stats: ColumnStats, num_rows: int, rule: CheckRule
    ) -> pd.Series:
        """Generate values only from the allowed set."""
        values = rule.values
        indices = self.rng.integers(0, len(values), size=num_rows)
        series = pd.Series([values[i] for i in indices], name=col_stats.name)

        if col_stats.nullable and col_stats.null_ratio > 0:
            mask = self.rng.random(num_rows) < col_stats.null_ratio
            series = series.where(~mask, other=None)

        return series

    def _generate_from_between(
        self, col_stats: ColumnStats, num_rows: int, rule: CheckRule
    ) -> pd.Series:
        """Generate values within BETWEEN bounds."""
        min_val = rule.min_value
        max_val = rule.max_value

        # Detect if it's a time value (string like 'HH:MM:SS')
        if isinstance(min_val, str) and ":" in min_val:
            return self._generate_time_between(col_stats, num_rows, min_val, max_val)

        # Numeric between
        if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
            if isinstance(min_val, int) and isinstance(max_val, int):
                values = self.rng.integers(min_val, max_val + 1, size=num_rows)
            else:
                values = self.rng.uniform(float(min_val), float(max_val), size=num_rows)
            series = pd.Series(values, name=col_stats.name)
            if col_stats.nullable and col_stats.null_ratio > 0:
                mask = self.rng.random(num_rows) < col_stats.null_ratio
                series = series.where(~mask, other=None)
            return series

        # Fallback to regular generation
        return self._generate_column(col_stats, num_rows)

    def _generate_time_between(
        self, col_stats: ColumnStats, num_rows: int, min_time: str, max_time: str
    ) -> pd.Series:
        """Generate time values within a range."""
        # Parse HH:MM:SS to total seconds
        def time_to_seconds(t: str) -> int:
            parts = t.split(":")
            h, m = int(parts[0]), int(parts[1])
            s = int(parts[2]) if len(parts) > 2 else 0
            return h * 3600 + m * 60 + s

        min_sec = time_to_seconds(min_time)
        max_sec = time_to_seconds(max_time)

        random_seconds = self.rng.integers(min_sec, max_sec + 1, size=num_rows)
        times = []
        for sec in random_seconds:
            h = sec // 3600
            m = (sec % 3600) // 60
            s = sec % 60
            times.append(f"{h:02d}:{m:02d}:{s:02d}")

        series = pd.Series(times, name=col_stats.name)
        if col_stats.nullable and col_stats.null_ratio > 0:
            mask = self.rng.random(num_rows) < col_stats.null_ratio
            series = series.where(~mask, other=None)
        return series

    def _generate_from_like(
        self, col_stats: ColumnStats, num_rows: int, rule: CheckRule
    ) -> pd.Series:
        """Generate strings matching a LIKE pattern."""
        pattern = rule.pattern
        if not pattern:
            return self._generate_column(col_stats, num_rows)

        values = [self._generate_string_from_like(pattern) for _ in range(num_rows)]
        series = pd.Series(values, name=col_stats.name)

        if col_stats.nullable and col_stats.null_ratio > 0:
            mask = self.rng.random(num_rows) < col_stats.null_ratio
            series = series.where(~mask, other=None)
        return series

    def _generate_string_from_like(self, pattern: str) -> str:
        """Generate a single string matching a SQL LIKE pattern.

        Supports:
        - Literal characters
        - % = any string (generates 0-5 random chars)
        - _ = single character
        - [0-9a-fA-F] = character class
        """
        result = []
        i = 0
        while i < len(pattern):
            ch = pattern[i]
            if ch == "%":
                # Generate 0-5 random alphanumeric chars
                length = int(self.rng.integers(0, 6))
                chars = "abcdefghijklmnopqrstuvwxyz0123456789"
                result.append("".join(self.rng.choice(list(chars), size=length)))
                i += 1
            elif ch == "_":
                chars = "abcdefghijklmnopqrstuvwxyz0123456789"
                result.append(self.rng.choice(list(chars)))
                i += 1
            elif ch == "[":
                # Character class: [0-9a-fA-F]
                end = pattern.find("]", i)
                if end == -1:
                    result.append(ch)
                    i += 1
                    continue
                char_class = pattern[i + 1:end]
                expanded = self._expand_char_class(char_class)
                if expanded:
                    result.append(self.rng.choice(list(expanded)))
                i = end + 1
            else:
                result.append(ch)
                i += 1
        return "".join(result)

    def _expand_char_class(self, char_class: str) -> str:
        """Expand a SQL character class like '0-9a-fA-F' into all valid chars."""
        result = []
        i = 0
        while i < len(char_class):
            if i + 2 < len(char_class) and char_class[i + 1] == "-":
                start_char = char_class[i]
                end_char = char_class[i + 2]
                for c in range(ord(start_char), ord(end_char) + 1):
                    result.append(chr(c))
                i += 3
            else:
                result.append(char_class[i])
                i += 1
        return "".join(result)

    def _generate_with_comparisons(
        self, col_stats: ColumnStats, num_rows: int, rules: List[CheckRule]
    ) -> pd.Series:
        """Generate values satisfying comparison constraints (>=, <=, >, <)."""
        # Start with the column's declared min/max
        effective_min = col_stats.min_value
        effective_max = col_stats.max_value

        for rule in rules:
            if rule.operator in (">=", ">") and rule.min_value is not None:
                val = rule.min_value
                if rule.operator == ">":
                    val = val + 1 if isinstance(val, int) else val + 0.01
                if effective_min is None or val > effective_min:
                    effective_min = val
            elif rule.operator in ("<=", "<") and rule.max_value is not None:
                val = rule.max_value
                if rule.operator == "<":
                    val = val - 1 if isinstance(val, int) else val - 0.01
                if effective_max is None or val < effective_max:
                    effective_max = val

        # Create a temporary ColumnStats with adjusted bounds
        adjusted = ColumnStats(
            name=col_stats.name,
            data_type=col_stats.data_type,
            nullable=col_stats.nullable,
            distinct_count=col_stats.distinct_count,
            min_value=effective_min,
            max_value=effective_max,
            null_count=col_stats.null_count,
            avg_length=col_stats.avg_length,
            max_length=col_stats.max_length,
            is_unique=col_stats.is_unique,
        )
        return self._generate_column(adjusted, num_rows)

    def _filter_not_like(self, series: pd.Series, rules: List[CheckRule]) -> pd.Series:
        """Filter out values containing forbidden substrings."""
        forbidden = [r.pattern.lower() for r in rules if r.pattern]
        if not forbidden:
            return series

        def is_valid(val):
            if pd.isna(val):
                return True
            val_lower = str(val).lower()
            return not any(f in val_lower for f in forbidden)

        mask = series.apply(is_valid)
        # Replace invalid values with regenerated strings
        invalid_count = (~mask).sum()
        if invalid_count > 0:
            # Generate simple replacement strings that won't contain the forbidden words
            replacements = [
                "".join(self.rng.choice(list("abcdefghijklmnopqrstuvwxyz"), size=10))
                for _ in range(invalid_count)
            ]
            series = series.copy()
            series.loc[~mask] = replacements

        return series

    def _enforce_cross_column_checks(
        self, df: pd.DataFrame, rules: List[CheckRule], table_stats: TableStats, num_rows: int
    ) -> pd.DataFrame:
        """Fix rows that violate cross-column CHECK constraints."""
        for rule in rules:
            if not rule.expression:
                continue
            # Handle NOT (A = 1 AND B = 1) pattern: flip one of the columns for violating rows
            # Parse: NOT (colA = val1 AND colB = val2)
            import re
            not_match = re.match(
                r"NOT\s*\(\s*\[?(\w+)\]?\s*=\s*(\d+)\s+AND\s+\[?(\w+)\]?\s*=\s*(\d+)\s*\)",
                rule.expression, re.IGNORECASE,
            )
            if not_match:
                col_a = not_match.group(1)
                val_a = int(not_match.group(2))
                col_b = not_match.group(3)
                val_b = int(not_match.group(4))

                # Find the actual column names (case-insensitive match)
                actual_a = next((c for c in df.columns if c.lower() == col_a.lower()), col_a)
                actual_b = next((c for c in df.columns if c.lower() == col_b.lower()), col_b)

                if actual_a in df.columns and actual_b in df.columns:
                    violating = (df[actual_a] == val_a) & (df[actual_b] == val_b)
                    if violating.any():
                        # Flip col_b to 0 for violating rows
                        df.loc[violating, actual_b] = 0

        return df

    def _generate_fk_column(
        self, parent_values: pd.Series, col_stats: Optional[ColumnStats], num_rows: int
    ) -> pd.Series:
        """Generate FK column by sampling from parent table's referenced column."""
        available = parent_values.dropna().values
        if len(available) == 0:
            # Fallback: generate integers
            return pd.Series(self.rng.integers(1, num_rows + 1, size=num_rows))

        # Sample with replacement from parent values
        indices = self.rng.integers(0, len(available), size=num_rows)
        values = available[indices]
        series = pd.Series(values)

        # Apply null injection if the FK column is nullable
        if col_stats and col_stats.nullable and col_stats.null_ratio > 0:
            mask = self.rng.random(num_rows) < col_stats.null_ratio
            series = series.where(~mask, other=None)

        return series

    def _enforce_unique(
        self, df: pd.DataFrame, unique_cols: List[str], table_stats: TableStats
    ) -> pd.DataFrame:
        """Ensure a set of columns has no duplicate combinations."""
        if not all(col in df.columns for col in unique_cols):
            return df

        # Check for duplicates
        duplicated = df.duplicated(subset=unique_cols, keep="first")
        if not duplicated.any():
            return df

        # For single-column FK+unique: sample without replacement from parent values
        if len(unique_cols) == 1:
            col_name = unique_cols[0]
            fk = table_stats.get_fk_for_column(col_name)
            if fk:
                parent_df = self._generated_tables.get(fk.parent_full_name)
                if parent_df is not None:
                    parent_col = fk.parent_columns[fk.child_columns.index(col_name)]
                    available = parent_df[parent_col].dropna().unique()
                    n_needed = len(df)
                    if len(available) >= n_needed:
                        sampled = self.rng.choice(available, size=n_needed, replace=False)
                        df[col_name] = sampled
                        return df

        # Regenerate duplicated rows until unique (with retry limit)
        max_retries = 5
        for _ in range(max_retries):
            dup_mask = df.duplicated(subset=unique_cols, keep="first")
            if not dup_mask.any():
                break

            num_dups = dup_mask.sum()
            for col_name in unique_cols:
                col_stats = table_stats.get_column(col_name)
                if col_stats:
                    new_values = self._generate_column(col_stats, num_dups)
                    df.loc[dup_mask, col_name] = new_values.values

        # Final fallback: append suffix to make unique for string columns
        dup_mask = df.duplicated(subset=unique_cols, keep="first")
        if dup_mask.any():
            for col_name in unique_cols:
                col_stats = table_stats.get_column(col_name)
                if col_stats and ("char" in col_stats.data_type.lower() or "text" in col_stats.data_type.lower()):
                    dup_indices = df.index[dup_mask]
                    for i, idx in enumerate(dup_indices):
                        df.at[idx, col_name] = f"{df.at[idx, col_name]}_{i}"
                elif col_stats:
                    # For numeric: offset duplicates
                    dup_indices = df.index[dup_mask]
                    for i, idx in enumerate(dup_indices):
                        df.at[idx, col_name] = df.at[idx, col_name] + i + 1

        return df

    def _generate_column(self, col_stats: ColumnStats, num_rows: int) -> pd.Series:
        dtype = col_stats.data_type.lower()

        if any(t in dtype for t in ("int", "serial", "bigint", "smallint", "tinyint")):
            values = self._generate_integer(col_stats, num_rows)
        elif any(t in dtype for t in ("double", "float", "real", "numeric", "decimal")):
            values = self._generate_numeric(col_stats, num_rows)
        elif any(t in dtype for t in ("char", "varchar", "text", "string", "clob")):
            values = self._generate_string(col_stats, num_rows)
        elif "date" in dtype and "time" not in dtype:
            values = self._generate_date(col_stats, num_rows)
        elif "timestamp" in dtype or "datetime" in dtype:
            values = self._generate_timestamp(col_stats, num_rows)
        elif "bool" in dtype:
            values = self._generate_boolean(num_rows)
        elif "blob" in dtype or "binary" in dtype or "bytea" in dtype:
            values = self._generate_binary(col_stats, num_rows)
        elif "uuid" in dtype or "uniqueidentifier" in dtype:
            values = self._generate_uuid(num_rows)
        else:
            values = self._generate_string(col_stats, num_rows)

        series = pd.Series(values, name=col_stats.name)

        # Inject nulls
        if col_stats.nullable and col_stats.null_ratio > 0:
            mask = self.rng.random(num_rows) < col_stats.null_ratio
            series = series.where(~mask, other=None)

        return series

    def _generate_integer(self, col_stats: ColumnStats, n: int) -> np.ndarray:
        min_val = int(col_stats.min_value) if col_stats.min_value is not None else 0
        max_val = int(col_stats.max_value) if col_stats.max_value is not None else 1_000_000

        # When unique, sample without replacement from available range
        if col_stats.is_unique:
            range_size = max_val - min_val + 1
            if range_size >= n:
                # Sample n unique values from the full range
                all_vals = np.arange(min_val, max_val + 1)
                return self.rng.choice(all_vals, size=n, replace=False)
            else:
                # Range too small, extend it
                return np.arange(min_val, min_val + n)

        if col_stats.distinct_count and col_stats.distinct_count < 50:
            # Low cardinality - pick from discrete set
            categories = np.arange(min_val, min_val + col_stats.distinct_count)
            return self.rng.choice(categories, size=n)

        return self.rng.integers(min_val, max_val + 1, size=n)

    def _generate_numeric(self, col_stats: ColumnStats, n: int) -> np.ndarray:
        min_val = float(col_stats.min_value) if col_stats.min_value is not None else 0.0
        max_val = float(col_stats.max_value) if col_stats.max_value is not None else 1_000_000.0
        return self.rng.uniform(min_val, max_val, size=n)

    def _generate_string(self, col_stats: ColumnStats, n: int) -> List[str]:
        if col_stats.is_categorical and col_stats.distinct_count:
            categories = [f"Category_{i}" for i in range(col_stats.distinct_count)]
            indices = self.rng.integers(0, len(categories), size=n)
            return [categories[i] for i in indices]

        target_length = int(col_stats.avg_length) if col_stats.avg_length else 12
        target_length = max(4, min(target_length, 200))

        # Check if it looks like an ID/GUID column
        name_lower = col_stats.name.lower()
        if any(k in name_lower for k in ("guid", "uuid", "id")):
            return [str(uuid.uuid4()) for _ in range(n)]

        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        return [
            "".join(self.rng.choice(list(chars), size=target_length))
            for _ in range(n)
        ]

    def _generate_date(self, col_stats: ColumnStats, n: int) -> List:
        try:
            min_date = pd.to_datetime(col_stats.min_value) if col_stats.min_value else datetime(2020, 1, 1)
            max_date = pd.to_datetime(col_stats.max_value) if col_stats.max_value else datetime.now()
        except (ValueError, TypeError):
            min_date = datetime(2020, 1, 1)
            max_date = datetime.now()

        days_range = (max_date - min_date).days
        if days_range <= 0:
            days_range = 365 * 5

        random_days = self.rng.integers(0, days_range, size=n)
        return [min_date + timedelta(days=int(d)) for d in random_days]

    def _generate_timestamp(self, col_stats: ColumnStats, n: int) -> List:
        try:
            min_ts = pd.to_datetime(col_stats.min_value) if col_stats.min_value else datetime(2020, 1, 1)
            max_ts = pd.to_datetime(col_stats.max_value) if col_stats.max_value else datetime.now()
        except (ValueError, TypeError):
            min_ts = datetime(2020, 1, 1)
            max_ts = datetime.now()

        seconds_range = int((max_ts - min_ts).total_seconds())
        if seconds_range <= 0:
            seconds_range = 365 * 5 * 86400

        random_seconds = self.rng.integers(0, seconds_range, size=n)
        return [min_ts + timedelta(seconds=int(s)) for s in random_seconds]

    def _generate_boolean(self, n: int) -> np.ndarray:
        return self.rng.choice([True, False], size=n)

    def _generate_binary(self, col_stats: ColumnStats, n: int) -> List[bytes]:
        length = int(col_stats.avg_length) if col_stats.avg_length else 16
        return [self.rng.bytes(length) for _ in range(n)]

    def _generate_uuid(self, n: int) -> List[str]:
        return [str(uuid.uuid4()) for _ in range(n)]
