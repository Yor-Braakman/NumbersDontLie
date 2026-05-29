"""Tests for CHECK constraint parsing and constraint-aware data generation."""

import pytest

from numbers_dont_lie.check_constraints import CheckConstraintParser, CheckRule
from numbers_dont_lie.generator import SyntheticDataGenerator
from numbers_dont_lie.models import ColumnStats, Constraint, TableStats


# --- Parser Tests ---


class TestCheckParserInList:
    """Parse IN (value_list) constraints."""

    def test_integer_in_list(self):
        parser = CheckConstraintParser()
        rules = parser.parse("[StoryPoints] IN (1, 2, 3, 5, 8, 13, 21)", ["StoryPoints"])
        assert len(rules) == 1
        assert rules[0].rule_type == "in_list"
        assert rules[0].column == "StoryPoints"
        assert rules[0].values == [1, 2, 3, 5, 8, 13, 21]

    def test_in_list_with_parens(self):
        parser = CheckConstraintParser()
        # SQL Server wraps in extra parens: ([col] IN ((1),(2),(3)))
        rules = parser.parse("([StoryPoints] IN (1, 2, 3, 5, 8, 13, 21))", ["StoryPoints"])
        assert len(rules) == 1
        assert rules[0].values == [1, 2, 3, 5, 8, 13, 21]

    def test_string_in_list(self):
        parser = CheckConstraintParser()
        rules = parser.parse("[Status] IN ('active', 'inactive', 'pending')", ["Status"])
        assert len(rules) == 1
        assert rules[0].values == ["active", "inactive", "pending"]


class TestCheckParserBetween:
    """Parse BETWEEN constraints."""

    def test_time_between(self):
        parser = CheckConstraintParser()
        rules = parser.parse("[StartTime] BETWEEN '08:00:00' AND '18:00:00'", ["StartTime"])
        assert len(rules) == 1
        assert rules[0].rule_type == "between"
        assert rules[0].min_value == "08:00:00"
        assert rules[0].max_value == "18:00:00"

    def test_numeric_between(self):
        parser = CheckConstraintParser()
        rules = parser.parse("[Price] BETWEEN 0.01 AND 99999.99", ["Price"])
        assert len(rules) == 1
        assert rules[0].min_value == 0.01
        assert rules[0].max_value == 99999.99


class TestCheckParserLike:
    """Parse LIKE pattern constraints."""

    def test_hex_color_pattern(self):
        parser = CheckConstraintParser()
        expr = "[PrimaryColor] LIKE '#[0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F]'"
        rules = parser.parse(expr, ["PrimaryColor"])
        assert len(rules) == 1
        assert rules[0].rule_type == "like_pattern"
        assert rules[0].column == "PrimaryColor"
        assert "#" in rules[0].pattern

    def test_suffix_like_pattern(self):
        parser = CheckConstraintParser()
        rules = parser.parse("[Price] LIKE '%.99'", ["Price"])
        assert len(rules) == 1
        assert rules[0].rule_type == "like_pattern"
        assert rules[0].pattern == "%.99"


class TestCheckParserNotLike:
    """Parse NOT LIKE constraints."""

    def test_single_not_like(self):
        parser = CheckConstraintParser()
        rules = parser.parse("[PitchText] NOT LIKE '%synergy%'", ["PitchText"])
        assert len(rules) == 1
        assert rules[0].rule_type == "not_like"
        assert rules[0].pattern == "synergy"

    def test_multiple_not_like(self):
        parser = CheckConstraintParser()
        expr = (
            "[PitchText] NOT LIKE '%synergy%' AND "
            "[PitchText] NOT LIKE '%paradigm shift%' AND "
            "[PitchText] NOT LIKE '%outside the box%'"
        )
        rules = parser.parse(expr, ["PitchText"])
        assert len(rules) == 3
        assert all(r.rule_type == "not_like" for r in rules)
        patterns = [r.pattern for r in rules]
        assert "synergy" in patterns
        assert "paradigm shift" in patterns
        assert "outside the box" in patterns


class TestCheckParserCrossColumn:
    """Parse cross-column constraints."""

    def test_mutual_exclusion(self):
        parser = CheckConstraintParser()
        expr = "NOT (IsBeta = 1 AND IsEnterpriseOnly = 1)"
        rules = parser.parse(expr, ["IsBeta", "IsEnterpriseOnly"])
        assert len(rules) == 1
        assert rules[0].rule_type == "cross_column"
        assert "IsBeta" in rules[0].related_columns
        assert "IsEnterpriseOnly" in rules[0].related_columns


class TestCheckParserComparisons:
    """Parse comparison constraints."""

    def test_greater_than_or_equal(self):
        parser = CheckConstraintParser()
        rules = parser.parse("[Price] >= 100.00", ["Price"])
        assert len(rules) == 1
        assert rules[0].rule_type == "comparison"
        assert rules[0].operator == ">="
        assert rules[0].min_value == 100.0

    def test_less_than(self):
        parser = CheckConstraintParser()
        rules = parser.parse("[Age] < 150", ["Age"])
        assert len(rules) == 1
        assert rules[0].operator == "<"
        assert rules[0].max_value == 150


# --- Generator Tests ---


class TestGeneratorInListConstraint:
    """Generator respects IN-list CHECK constraints."""

    def test_fibonacci_only(self):
        """StoryPoints IN (1, 2, 3, 5, 8, 13, 21) - only these values generated."""
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)

        fibonacci = [1, 2, 3, 5, 8, 13, 21]
        stats = TableStats(
            database_name="db", schema_name="dbo", table_name="AgileTasks",
            num_records=1000,
            columns=[
                ColumnStats(name="TaskID", data_type="integer", nullable=False, distinct_count=1000, min_value=1, max_value=1000, is_unique=True),
                ColumnStats(name="TaskName", data_type="varchar", nullable=False, distinct_count=500, avg_length=20.0),
                ColumnStats(name="StoryPoints", data_type="integer", nullable=False, distinct_count=7, min_value=1, max_value=21),
            ],
            constraints=[
                Constraint(name="pk", constraint_type="PRIMARY KEY", columns=["TaskID"]),
                Constraint(
                    name="CK_Fibonacci_Only",
                    constraint_type="CHECK",
                    columns=["StoryPoints"],
                    check_expression="[StoryPoints] IN (1, 2, 3, 5, 8, 13, 21)",
                ),
            ],
        )
        df = gen.generate(stats)
        unique_vals = set(df["StoryPoints"].unique())
        assert unique_vals.issubset(set(fibonacci)), f"Invalid values: {unique_vals - set(fibonacci)}"
        assert len(unique_vals) > 1  # Should use multiple values

    def test_in_list_distribution(self):
        """Values from IN list should be roughly uniformly distributed."""
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)

        stats = TableStats(
            database_name="db", schema_name="s", table_name="t", num_records=7000,
            columns=[
                ColumnStats(name="val", data_type="integer", nullable=False, distinct_count=7, min_value=1, max_value=21),
            ],
            constraints=[
                Constraint(
                    name="ck", constraint_type="CHECK", columns=["val"],
                    check_expression="[val] IN (1, 2, 3, 5, 8, 13, 21)",
                ),
            ],
        )
        df = gen.generate(stats)
        counts = df["val"].value_counts()
        expected = 7000 / 7  # ~1000 each
        for val, count in counts.items():
            assert count > expected * 0.7, f"Value {val} underrepresented: {count}"
            assert count < expected * 1.3, f"Value {val} overrepresented: {count}"


class TestGeneratorBetweenConstraint:
    """Generator respects BETWEEN CHECK constraints."""

    def test_time_between_business_hours(self):
        """StartTime BETWEEN '08:00:00' AND '18:00:00'."""
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)

        stats = TableStats(
            database_name="db", schema_name="dbo", table_name="MaintenanceWindows",
            num_records=500,
            columns=[
                ColumnStats(name="WindowID", data_type="integer", nullable=False, distinct_count=500, min_value=1, max_value=500, is_unique=True),
                ColumnStats(name="TechnicianName", data_type="varchar", nullable=False, distinct_count=50, avg_length=15.0),
                ColumnStats(name="StartTime", data_type="time", nullable=False, distinct_count=200),
            ],
            constraints=[
                Constraint(name="pk", constraint_type="PRIMARY KEY", columns=["WindowID"]),
                Constraint(
                    name="CK_Core_Business_Hours",
                    constraint_type="CHECK",
                    columns=["StartTime"],
                    check_expression="[StartTime] BETWEEN '08:00:00' AND '18:00:00'",
                ),
            ],
        )
        df = gen.generate(stats)

        for time_val in df["StartTime"]:
            assert time_val >= "08:00:00", f"Time {time_val} is before 08:00:00"
            assert time_val <= "18:00:00", f"Time {time_val} is after 18:00:00"

    def test_numeric_between(self):
        """Price BETWEEN 10.00 AND 500.00."""
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)

        stats = TableStats(
            database_name="db", schema_name="s", table_name="t", num_records=1000,
            columns=[
                ColumnStats(name="Price", data_type="decimal", nullable=False, distinct_count=500, min_value=10.0, max_value=500.0),
            ],
            constraints=[
                Constraint(
                    name="ck", constraint_type="CHECK", columns=["Price"],
                    check_expression="[Price] BETWEEN 10.00 AND 500.00",
                ),
            ],
        )
        df = gen.generate(stats)
        assert df["Price"].min() >= 10.0
        assert df["Price"].max() <= 500.0


class TestGeneratorLikeConstraint:
    """Generator respects LIKE pattern CHECK constraints."""

    def test_hex_color_pattern(self):
        """PrimaryColor LIKE '#[0-9a-fA-F][0-9a-fA-F]...' generates valid hex colors."""
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)

        stats = TableStats(
            database_name="db", schema_name="dbo", table_name="UIThemes",
            num_records=100,
            columns=[
                ColumnStats(name="ThemeID", data_type="integer", nullable=False, distinct_count=100, min_value=1, max_value=100, is_unique=True),
                ColumnStats(name="ThemeName", data_type="varchar", nullable=False, distinct_count=100, avg_length=12.0),
                ColumnStats(name="PrimaryColor", data_type="char", nullable=False, distinct_count=100, avg_length=7.0),
            ],
            constraints=[
                Constraint(name="pk", constraint_type="PRIMARY KEY", columns=["ThemeID"]),
                Constraint(
                    name="CK_Valid_Hex_Color",
                    constraint_type="CHECK",
                    columns=["PrimaryColor"],
                    check_expression="[PrimaryColor] LIKE '#[0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F]'",
                ),
            ],
        )
        df = gen.generate(stats)

        import re
        hex_pattern = re.compile(r"^#[0-9a-fA-F]{6}$")
        for color in df["PrimaryColor"]:
            assert hex_pattern.match(color), f"Invalid hex color: {color}"

    def test_like_generates_correct_length(self):
        """Generated values match the expected length from the pattern."""
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)

        stats = TableStats(
            database_name="db", schema_name="s", table_name="t", num_records=100,
            columns=[
                ColumnStats(name="code", data_type="char", nullable=False, distinct_count=100, avg_length=7.0),
            ],
            constraints=[
                Constraint(
                    name="ck", constraint_type="CHECK", columns=["code"],
                    check_expression="[code] LIKE '#[0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F]'",
                ),
            ],
        )
        df = gen.generate(stats)
        # All values should be exactly 7 chars: # + 6 hex digits
        assert all(len(v) == 7 for v in df["code"])


class TestGeneratorNotLikeConstraint:
    """Generator respects NOT LIKE constraints (no buzzwords)."""

    def test_no_forbidden_substrings(self):
        """PitchText NOT LIKE '%synergy%' etc. - forbidden words never appear."""
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)

        stats = TableStats(
            database_name="db", schema_name="dbo", table_name="MarketingCopy",
            num_records=500,
            columns=[
                ColumnStats(name="CampaignID", data_type="integer", nullable=False, distinct_count=500, min_value=1, max_value=500, is_unique=True),
                ColumnStats(name="PitchText", data_type="varchar", nullable=False, distinct_count=500, avg_length=100.0),
            ],
            constraints=[
                Constraint(name="pk", constraint_type="PRIMARY KEY", columns=["CampaignID"]),
                Constraint(
                    name="CK_No_Synergy_Buzzwords",
                    constraint_type="CHECK",
                    columns=["PitchText"],
                    check_expression=(
                        "[PitchText] NOT LIKE '%synergy%' AND "
                        "[PitchText] NOT LIKE '%paradigm shift%' AND "
                        "[PitchText] NOT LIKE '%outside the box%' AND "
                        "[PitchText] NOT LIKE '%utilize%'"
                    ),
                ),
            ],
        )
        df = gen.generate(stats)

        forbidden = ["synergy", "paradigm shift", "outside the box", "utilize"]
        for text in df["PitchText"]:
            text_lower = text.lower()
            for word in forbidden:
                assert word not in text_lower, f"Found forbidden '{word}' in: {text}"


class TestGeneratorCrossColumnConstraint:
    """Generator respects cross-column CHECK constraints."""

    def test_mutually_exclusive_flags(self):
        """NOT (IsBeta = 1 AND IsEnterpriseOnly = 1) - never both true."""
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)

        stats = TableStats(
            database_name="db", schema_name="dbo", table_name="FeatureFlags",
            num_records=1000,
            columns=[
                ColumnStats(name="FeatureID", data_type="integer", nullable=False, distinct_count=1000, min_value=1, max_value=1000, is_unique=True),
                ColumnStats(name="FeatureName", data_type="varchar", nullable=False, distinct_count=500, avg_length=20.0),
                ColumnStats(name="IsBeta", data_type="bit", nullable=False, distinct_count=2, min_value=0, max_value=1),
                ColumnStats(name="IsEnterpriseOnly", data_type="bit", nullable=False, distinct_count=2, min_value=0, max_value=1),
            ],
            constraints=[
                Constraint(name="pk", constraint_type="PRIMARY KEY", columns=["FeatureID"]),
                Constraint(
                    name="CK_Mutually_Exclusive_Toggles",
                    constraint_type="CHECK",
                    columns=["IsBeta", "IsEnterpriseOnly"],
                    check_expression="NOT (IsBeta = 1 AND IsEnterpriseOnly = 1)",
                ),
            ],
        )
        df = gen.generate(stats)

        # No row should have both = 1
        both_true = ((df["IsBeta"] == 1) & (df["IsEnterpriseOnly"] == 1)).sum()
        assert both_true == 0, f"Found {both_true} rows where both flags are 1"


class TestGeneratorComparisonConstraint:
    """Generator respects comparison (>=, <=) constraints."""

    def test_minimum_price(self):
        """Price >= 100.00."""
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)

        stats = TableStats(
            database_name="db", schema_name="s", table_name="t", num_records=1000,
            columns=[
                ColumnStats(name="Price", data_type="decimal", nullable=False, distinct_count=500, min_value=100.0, max_value=9999.99),
            ],
            constraints=[
                Constraint(
                    name="ck", constraint_type="CHECK", columns=["Price"],
                    check_expression="[Price] >= 100.00",
                ),
            ],
        )
        df = gen.generate(stats)
        assert df["Price"].min() >= 100.0, f"Min price: {df['Price'].min()}"


class TestGeneratorNoCheckConstraint:
    """Tables without CHECK constraints still generate normally."""

    def test_no_constraints_works(self):
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)

        stats = TableStats(
            database_name="db", schema_name="s", table_name="t", num_records=100,
            columns=[
                ColumnStats(name="id", data_type="integer", nullable=False, distinct_count=100, min_value=1, max_value=100),
                ColumnStats(name="name", data_type="varchar", nullable=False, distinct_count=80, avg_length=10.0),
            ],
        )
        df = gen.generate(stats)
        assert len(df) == 100
        assert "id" in df.columns
        assert "name" in df.columns
