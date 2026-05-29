"""CHECK constraint parser and value generator.

Parses SQL CHECK expressions into generation rules that the synthetic data
generator can use to produce conforming values.
"""

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class CheckRule:
    """A parsed rule derived from a CHECK constraint expression."""

    column: str
    rule_type: str  # "in_list", "between", "like_pattern", "not_like", "comparison", "cross_column"
    values: List[Any] = field(default_factory=list)  # For in_list
    min_value: Optional[Any] = None  # For between/comparison
    max_value: Optional[Any] = None  # For between/comparison
    pattern: Optional[str] = None  # For like/not_like (converted to generation hint)
    operator: Optional[str] = None  # For comparison: >=, <=, >, <, =, !=
    # For cross_column rules
    related_columns: List[str] = field(default_factory=list)
    expression: Optional[str] = None  # Original sub-expression for complex rules


class CheckConstraintParser:
    """Parse CHECK constraint expressions into generation rules.

    Supports common patterns:
    - IN (value_list)
    - BETWEEN min AND max
    - column >= val, column <= val, etc.
    - LIKE 'pattern'
    - NOT LIKE 'pattern'
    - Cross-column constraints (NOT (A = 1 AND B = 1))
    """

    def parse(self, expression: str, columns: List[str]) -> List[CheckRule]:
        """Parse a CHECK expression into a list of CheckRules.

        Args:
            expression: The SQL CHECK expression (without the CHECK keyword).
            columns: List of column names in the table for reference.

        Returns:
            List of CheckRule objects. May be empty if expression is unparseable.
        """
        if not expression:
            return []

        # Strip outer parens that SQL Server adds: ([col]>=(0))
        expr = self._strip_outer_parens(expression)

        rules = []

        # Try each pattern in order of specificity
        rules.extend(self._parse_in_list(expr, columns))
        if rules:
            return rules

        rules.extend(self._parse_between(expr, columns))
        if rules:
            return rules

        rules.extend(self._parse_not_like(expr, columns))
        if rules:
            return rules

        rules.extend(self._parse_like(expr, columns))
        if rules:
            return rules

        rules.extend(self._parse_cross_column(expr, columns))
        if rules:
            return rules

        rules.extend(self._parse_comparisons(expr, columns))
        return rules

    def _strip_outer_parens(self, expr: str) -> str:
        """Strip wrapping parentheses."""
        expr = expr.strip()
        while expr.startswith("(") and expr.endswith(")"):
            # Only strip if they're matching outer parens
            depth = 0
            matched = True
            for i, ch in enumerate(expr):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                if depth == 0 and i < len(expr) - 1:
                    matched = False
                    break
            if matched:
                expr = expr[1:-1].strip()
            else:
                break
        return expr

    def _find_column(self, token: str, columns: List[str]) -> Optional[str]:
        """Match a token to a column name (handles [brackets] and case)."""
        # Strip SQL Server brackets
        clean = token.strip().strip("[]").strip()
        for col in columns:
            if col.lower() == clean.lower():
                return col
        return None

    def _parse_in_list(self, expr: str, columns: List[str]) -> List[CheckRule]:
        """Parse: column IN (val1, val2, ...)"""
        # Pattern: [col] IN (values) or col IN (values)
        pattern = r"(\[?\w+\]?)\s+IN\s*\(([^)]+)\)"
        match = re.search(pattern, expr, re.IGNORECASE)
        if not match:
            return []

        col_token = match.group(1)
        values_str = match.group(2)

        col = self._find_column(col_token, columns)
        if not col:
            return []

        values = self._parse_value_list(values_str)
        if not values:
            return []

        return [CheckRule(column=col, rule_type="in_list", values=values)]

    def _parse_between(self, expr: str, columns: List[str]) -> List[CheckRule]:
        """Parse: column BETWEEN min AND max"""
        pattern = r"(\[?\w+\]?)\s+BETWEEN\s+(['\d:.\-]+)\s+AND\s+(['\d:.\-]+)"
        match = re.search(pattern, expr, re.IGNORECASE)
        if not match:
            return []

        col_token = match.group(1)
        min_val = self._parse_single_value(match.group(2))
        max_val = self._parse_single_value(match.group(3))

        col = self._find_column(col_token, columns)
        if not col:
            return []

        return [CheckRule(column=col, rule_type="between", min_value=min_val, max_value=max_val)]

    def _parse_not_like(self, expr: str, columns: List[str]) -> List[CheckRule]:
        """Parse: col NOT LIKE '%pattern%' AND col NOT LIKE '%pattern2%'"""
        pattern = r"(\[?\w+\]?)\s+NOT\s+LIKE\s+'([^']+)'"
        matches = re.findall(pattern, expr, re.IGNORECASE)
        if not matches:
            return []

        rules = []
        for col_token, like_pattern in matches:
            col = self._find_column(col_token, columns)
            if col:
                # Convert SQL LIKE to a forbidden substring
                forbidden = like_pattern.replace("%", "").replace("_", "")
                rules.append(CheckRule(
                    column=col, rule_type="not_like", pattern=forbidden, expression=like_pattern,
                ))

        return rules

    def _parse_like(self, expr: str, columns: List[str]) -> List[CheckRule]:
        """Parse: col LIKE 'pattern'"""
        # Exclude NOT LIKE (already handled)
        if "NOT LIKE" in expr.upper():
            return []

        pattern = r"(\[?\w+\]?)\s+LIKE\s+'([^']+)'"
        matches = re.findall(pattern, expr, re.IGNORECASE)
        if not matches:
            return []

        rules = []
        for col_token, like_pattern in matches:
            col = self._find_column(col_token, columns)
            if col:
                rules.append(CheckRule(
                    column=col, rule_type="like_pattern", pattern=like_pattern,
                ))

        return rules

    def _parse_cross_column(self, expr: str, columns: List[str]) -> List[CheckRule]:
        """Parse: NOT (colA = val AND colB = val) style mutual exclusion."""
        # Pattern: NOT (expr)
        not_match = re.match(r"NOT\s*\((.+)\)", expr, re.IGNORECASE | re.DOTALL)
        if not not_match:
            return []

        inner = not_match.group(1)
        # Find all column references
        referenced = []
        for col in columns:
            if col.lower() in inner.lower() or f"[{col}]" in inner:
                referenced.append(col)

        if len(referenced) >= 2:
            return [CheckRule(
                column=referenced[0],
                rule_type="cross_column",
                related_columns=referenced,
                expression=expr,
            )]

        return []

    def _parse_comparisons(self, expr: str, columns: List[str]) -> List[CheckRule]:
        """Parse: col >= val, col <= val, col > val, col < val"""
        pattern = r"(\[?\w+\]?)\s*(>=|<=|>|<|!=|=)\s*(['\d.\-]+)"
        matches = re.findall(pattern, expr, re.IGNORECASE)
        if not matches:
            return []

        rules = []
        for col_token, op, val_str in matches:
            col = self._find_column(col_token, columns)
            if not col:
                continue
            value = self._parse_single_value(val_str)

            if op in (">=", ">"):
                rules.append(CheckRule(column=col, rule_type="comparison", operator=op, min_value=value))
            elif op in ("<=", "<"):
                rules.append(CheckRule(column=col, rule_type="comparison", operator=op, max_value=value))
            elif op == "=":
                rules.append(CheckRule(column=col, rule_type="in_list", values=[value]))
            elif op == "!=":
                rules.append(CheckRule(column=col, rule_type="comparison", operator=op, min_value=value))

        return rules

    def _parse_value_list(self, values_str: str) -> List[Any]:
        """Parse a comma-separated list of values."""
        values = []
        for item in values_str.split(","):
            val = self._parse_single_value(item.strip())
            if val is not None:
                values.append(val)
        return values

    def _parse_single_value(self, val_str: str) -> Any:
        """Parse a single value from a SQL expression."""
        val_str = val_str.strip().strip("()")
        # String literal
        if val_str.startswith("'") and val_str.endswith("'"):
            return val_str[1:-1]
        # Integer
        try:
            return int(val_str)
        except ValueError:
            pass
        # Float
        try:
            return float(val_str)
        except ValueError:
            pass
        return val_str
