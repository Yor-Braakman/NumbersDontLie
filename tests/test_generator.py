"""Unit tests for the synthetic data generator."""

import numpy as np
import pandas as pd
import pytest

from numbers_dont_lie.generator import SyntheticDataGenerator
from numbers_dont_lie.models import ColumnStats, TableStats


@pytest.fixture
def generator():
    return SyntheticDataGenerator(scale_factor=1.0, seed=42)


@pytest.fixture
def simple_table_stats():
    return TableStats(
        database_name="testdb",
        schema_name="public",
        table_name="users",
        num_records=1000,
        columns=[
            ColumnStats(name="id", data_type="integer", nullable=False, distinct_count=1000, min_value=1, max_value=1000),
            ColumnStats(name="name", data_type="varchar", nullable=False, distinct_count=800, avg_length=12.0),
            ColumnStats(name="email", data_type="varchar", nullable=True, distinct_count=950, null_count=50, avg_length=25.0),
            ColumnStats(name="age", data_type="integer", nullable=True, distinct_count=80, min_value=18, max_value=99, null_count=20),
            ColumnStats(name="salary", data_type="double", nullable=True, distinct_count=500, min_value=30000.0, max_value=200000.0, null_count=100),
            ColumnStats(name="is_active", data_type="boolean", nullable=False, distinct_count=2),
            ColumnStats(name="created_at", data_type="timestamp", nullable=False, distinct_count=1000, min_value="2020-01-01", max_value="2025-12-31"),
            ColumnStats(name="birth_date", data_type="date", nullable=True, distinct_count=900, min_value="1950-01-01", max_value="2005-12-31", null_count=10),
        ],
    )


@pytest.fixture
def categorical_table_stats():
    return TableStats(
        database_name="testdb",
        schema_name="public",
        table_name="orders",
        num_records=5000,
        columns=[
            ColumnStats(name="order_id", data_type="integer", nullable=False, distinct_count=5000, min_value=1, max_value=5000),
            ColumnStats(name="status", data_type="varchar", nullable=False, distinct_count=5, null_count=200, avg_length=8.0),
            ColumnStats(name="region", data_type="text", nullable=False, distinct_count=10, null_count=400, avg_length=6.0),
            ColumnStats(name="amount", data_type="decimal", nullable=False, distinct_count=3000, min_value=1.99, max_value=9999.99),
        ],
    )


class TestGeneratorRowCount:
    def test_generates_correct_row_count(self, generator, simple_table_stats):
        df = generator.generate(simple_table_stats)
        assert len(df) == 1000

    def test_scale_factor_increases_rows(self, simple_table_stats):
        gen = SyntheticDataGenerator(scale_factor=2.0, seed=42)
        df = gen.generate(simple_table_stats)
        assert len(df) == 2000

    def test_scale_factor_decreases_rows(self, simple_table_stats):
        gen = SyntheticDataGenerator(scale_factor=0.5, seed=42)
        df = gen.generate(simple_table_stats)
        assert len(df) == 500

    def test_zero_rows_gets_minimum(self):
        stats = TableStats(
            database_name="db", schema_name="s", table_name="t", num_records=0,
            columns=[ColumnStats(name="col", data_type="integer", nullable=False)],
        )
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)
        df = gen.generate(stats)
        assert len(df) == 100  # minimum fallback


class TestGeneratorColumns:
    def test_all_columns_present(self, generator, simple_table_stats):
        df = generator.generate(simple_table_stats)
        expected_cols = {col.name for col in simple_table_stats.columns}
        assert set(df.columns) == expected_cols

    def test_integer_column_range(self, generator, simple_table_stats):
        df = generator.generate(simple_table_stats)
        assert df["id"].min() >= 1
        assert df["id"].max() <= 1000

    def test_numeric_column_range(self, generator, simple_table_stats):
        df = generator.generate(simple_table_stats)
        non_null = df["salary"].dropna()
        assert non_null.min() >= 30000.0
        assert non_null.max() <= 200000.0

    def test_boolean_column_values(self, generator, simple_table_stats):
        df = generator.generate(simple_table_stats)
        assert set(df["is_active"].unique()).issubset({True, False})

    def test_string_column_not_empty(self, generator, simple_table_stats):
        df = generator.generate(simple_table_stats)
        non_null = df["name"].dropna()
        assert all(len(str(v)) > 0 for v in non_null)


class TestGeneratorNulls:
    def test_nullable_column_has_nulls(self, generator, simple_table_stats):
        df = generator.generate(simple_table_stats)
        # email has null_ratio > 0, so should have some nulls
        assert df["email"].isna().sum() > 0

    def test_non_nullable_column_no_nulls(self, generator, simple_table_stats):
        df = generator.generate(simple_table_stats)
        assert df["id"].isna().sum() == 0
        assert df["is_active"].isna().sum() == 0

    def test_null_ratio_approximately_correct(self, simple_table_stats):
        # Use large dataset for statistical significance
        stats = TableStats(
            database_name="db", schema_name="s", table_name="t", num_records=10000,
            columns=[
                ColumnStats(name="col", data_type="integer", nullable=True, distinct_count=100, null_count=100, min_value=0, max_value=999),
            ],
        )
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)
        df = gen.generate(stats)
        actual_ratio = df["col"].isna().sum() / len(df)
        expected_ratio = 100 / (100 + 100)  # 0.5
        assert abs(actual_ratio - expected_ratio) < 0.05  # within 5%


class TestGeneratorCategorical:
    def test_categorical_column_limited_values(self, generator, categorical_table_stats):
        df = generator.generate(categorical_table_stats)
        # "status" has distinct_count=5, should produce limited categories
        unique_count = df["status"].nunique()
        assert unique_count <= 5

    def test_categorical_column_has_prefix(self, generator, categorical_table_stats):
        df = generator.generate(categorical_table_stats)
        # Categorical strings get "Category_" prefix
        assert all(str(v).startswith("Category_") for v in df["status"])


class TestGeneratorDateTypes:
    def test_date_column_produces_dates(self, generator, simple_table_stats):
        df = generator.generate(simple_table_stats)
        non_null = df["birth_date"].dropna()
        # All values should be convertible to datetime
        dates = pd.to_datetime(non_null)
        assert len(dates) > 0

    def test_timestamp_column_produces_timestamps(self, generator, simple_table_stats):
        df = generator.generate(simple_table_stats)
        timestamps = pd.to_datetime(df["created_at"])
        assert len(timestamps) == 1000


class TestGeneratorReproducibility:
    def test_same_seed_same_output(self, simple_table_stats):
        gen1 = SyntheticDataGenerator(scale_factor=1.0, seed=123)
        gen2 = SyntheticDataGenerator(scale_factor=1.0, seed=123)
        df1 = gen1.generate(simple_table_stats)
        df2 = gen2.generate(simple_table_stats)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seed_different_output(self, simple_table_stats):
        gen1 = SyntheticDataGenerator(scale_factor=1.0, seed=1)
        gen2 = SyntheticDataGenerator(scale_factor=1.0, seed=2)
        df1 = gen1.generate(simple_table_stats)
        df2 = gen2.generate(simple_table_stats)
        # At least some columns should differ
        assert not df1["id"].equals(df2["id"])


class TestGeneratorUUID:
    def test_uuid_like_column(self):
        stats = TableStats(
            database_name="db", schema_name="s", table_name="t", num_records=100,
            columns=[
                ColumnStats(name="user_guid", data_type="varchar", nullable=False, distinct_count=100, avg_length=36.0),
            ],
        )
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)
        df = gen.generate(stats)
        # UUID columns should produce UUID-formatted strings
        sample = df["user_guid"].iloc[0]
        assert len(sample) == 36
        assert sample.count("-") == 4


class TestGeneratorBinary:
    def test_binary_column_produces_bytes(self):
        stats = TableStats(
            database_name="db", schema_name="s", table_name="t", num_records=50,
            columns=[
                ColumnStats(name="data", data_type="binary", nullable=False, distinct_count=50, avg_length=16.0),
            ],
        )
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)
        df = gen.generate(stats)
        assert all(isinstance(v, bytes) for v in df["data"])


class TestGeneratorCardinality:
    """Tests that the generated data respects the specified distinct count."""

    def test_low_cardinality_integer_bounded(self):
        stats = TableStats(
            database_name="db", schema_name="s", table_name="t", num_records=10000,
            columns=[
                ColumnStats(name="status_code", data_type="integer", nullable=False, distinct_count=5, min_value=0, max_value=4),
            ],
        )
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)
        df = gen.generate(stats)
        # With distinct_count=5 and min=0, should produce at most 5 unique values
        assert df["status_code"].nunique() <= 5

    def test_low_cardinality_categorical_string(self):
        stats = TableStats(
            database_name="db", schema_name="s", table_name="t", num_records=10000,
            columns=[
                ColumnStats(name="color", data_type="varchar", nullable=False, distinct_count=8, null_count=500, avg_length=10.0),
            ],
        )
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)
        df = gen.generate(stats)
        non_null = df["color"].dropna()
        # Categorical generator uses distinct_count categories
        assert non_null.nunique() <= 8

    def test_high_cardinality_integer_many_unique(self):
        stats = TableStats(
            database_name="db", schema_name="s", table_name="t", num_records=10000,
            columns=[
                ColumnStats(name="user_id", data_type="bigint", nullable=False, distinct_count=10000, min_value=1, max_value=1000000),
            ],
        )
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)
        df = gen.generate(stats)
        # High cardinality should produce many distinct values (close to num_records for large range)
        assert df["user_id"].nunique() > 9000

    def test_cardinality_with_nulls_excluded(self):
        stats = TableStats(
            database_name="db", schema_name="s", table_name="t", num_records=10000,
            columns=[
                ColumnStats(name="category", data_type="varchar", nullable=True, distinct_count=3, null_count=100, avg_length=8.0),
            ],
        )
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)
        df = gen.generate(stats)
        non_null = df["category"].dropna()
        # Should have at most 3 distinct non-null values
        assert non_null.nunique() <= 3

    def test_boolean_cardinality_is_two(self):
        stats = TableStats(
            database_name="db", schema_name="s", table_name="t", num_records=10000,
            columns=[
                ColumnStats(name="flag", data_type="boolean", nullable=False, distinct_count=2),
            ],
        )
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)
        df = gen.generate(stats)
        assert df["flag"].nunique() == 2

    def test_uuid_cardinality_all_unique(self):
        stats = TableStats(
            database_name="db", schema_name="s", table_name="t", num_records=1000,
            columns=[
                ColumnStats(name="request_id", data_type="varchar", nullable=False, distinct_count=1000, avg_length=36.0),
            ],
        )
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)
        df = gen.generate(stats)
        # UUID generation should produce all unique values
        assert df["request_id"].nunique() == 1000


class TestGeneratorDistribution:
    """Tests that the synthetic data histogram matches the expected uniform distribution."""

    def test_integer_uniform_distribution_across_bins(self):
        """Verify integers are uniformly distributed across the min-max range."""
        stats = TableStats(
            database_name="db", schema_name="s", table_name="t", num_records=100000,
            columns=[
                ColumnStats(name="value", data_type="integer", nullable=False, distinct_count=100000, min_value=0, max_value=1000),
            ],
        )
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)
        df = gen.generate(stats)

        # Split range into 10 equal bins and check each has roughly 10% of data
        bins = pd.cut(df["value"], bins=10)
        bin_counts = bins.value_counts()

        expected_per_bin = 100000 / 10
        for count in bin_counts:
            # Each bin should have between 8% and 12% (generous tolerance)
            assert count > expected_per_bin * 0.8, f"Bin underpopulated: {count}"
            assert count < expected_per_bin * 1.2, f"Bin overpopulated: {count}"

    def test_float_uniform_distribution_across_bins(self):
        """Verify floats are uniformly distributed across the min-max range."""
        stats = TableStats(
            database_name="db", schema_name="s", table_name="t", num_records=100000,
            columns=[
                ColumnStats(name="amount", data_type="double", nullable=False, distinct_count=100000, min_value=0.0, max_value=1000.0),
            ],
        )
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)
        df = gen.generate(stats)

        bins = pd.cut(df["amount"], bins=10)
        bin_counts = bins.value_counts()

        expected_per_bin = 100000 / 10
        for count in bin_counts:
            assert count > expected_per_bin * 0.8
            assert count < expected_per_bin * 1.2

    def test_date_uniform_distribution_across_months(self):
        """Verify dates are spread across the full date range."""
        stats = TableStats(
            database_name="db", schema_name="s", table_name="t", num_records=50000,
            columns=[
                ColumnStats(name="event_date", data_type="date", nullable=False, distinct_count=50000, min_value="2020-01-01", max_value="2024-12-31"),
            ],
        )
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)
        df = gen.generate(stats)

        dates = pd.to_datetime(df["event_date"])
        # Check that all 5 years are represented
        years = dates.dt.year.unique()
        assert set(range(2020, 2025)).issubset(set(years))

        # Check year distribution is roughly uniform (5 years = ~20% each)
        year_counts = dates.dt.year.value_counts()
        expected_per_year = 50000 / 5
        for year, count in year_counts.items():
            assert count > expected_per_year * 0.7, f"Year {year} underpopulated: {count}"
            assert count < expected_per_year * 1.3, f"Year {year} overpopulated: {count}"

    def test_categorical_uniform_across_categories(self):
        """Verify categorical values are uniformly distributed across categories."""
        stats = TableStats(
            database_name="db", schema_name="s", table_name="t", num_records=100000,
            columns=[
                ColumnStats(name="region", data_type="varchar", nullable=False, distinct_count=4, null_count=5000, avg_length=10.0),
            ],
        )
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)
        df = gen.generate(stats)

        non_null = df["region"].dropna()
        cat_counts = non_null.value_counts()

        # 4 categories should each have ~25% of non-null values
        expected_per_cat = len(non_null) / 4
        for cat, count in cat_counts.items():
            assert count > expected_per_cat * 0.75, f"Category '{cat}' underpopulated: {count}"
            assert count < expected_per_cat * 1.25, f"Category '{cat}' overpopulated: {count}"

    def test_integer_no_values_outside_range(self):
        """Verify no generated integers fall outside min-max bounds."""
        stats = TableStats(
            database_name="db", schema_name="s", table_name="t", num_records=50000,
            columns=[
                ColumnStats(name="score", data_type="integer", nullable=False, distinct_count=50000, min_value=100, max_value=500),
            ],
        )
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)
        df = gen.generate(stats)

        assert df["score"].min() >= 100
        assert df["score"].max() <= 500

    def test_float_no_values_outside_range(self):
        """Verify no generated floats fall outside min-max bounds."""
        stats = TableStats(
            database_name="db", schema_name="s", table_name="t", num_records=50000,
            columns=[
                ColumnStats(name="temperature", data_type="float", nullable=False, distinct_count=50000, min_value=-20.0, max_value=45.0),
            ],
        )
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)
        df = gen.generate(stats)

        assert df["temperature"].min() >= -20.0
        assert df["temperature"].max() <= 45.0

    def test_histogram_bin_edges_align_with_range(self):
        """Verify the histogram covers the full range with no gaps."""
        stats = TableStats(
            database_name="db", schema_name="s", table_name="t", num_records=100000,
            columns=[
                ColumnStats(name="measurement", data_type="double", nullable=False, distinct_count=100000, min_value=10.0, max_value=50.0),
            ],
        )
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)
        df = gen.generate(stats)

        # Create histogram with 20 bins
        counts, bin_edges = np.histogram(df["measurement"], bins=20)

        # No bin should be empty (with 100k rows and 20 bins, each should have ~5000)
        assert all(c > 0 for c in counts), f"Empty bins found: {counts}"

        # First bin edge should be near min, last near max
        assert bin_edges[0] >= 10.0
        assert bin_edges[-1] <= 50.0

    def test_narrow_range_integer_fills_all_values(self):
        """When range is small, all possible values should appear."""
        stats = TableStats(
            database_name="db", schema_name="s", table_name="t", num_records=10000,
            columns=[
                ColumnStats(name="rating", data_type="integer", nullable=False, distinct_count=5, min_value=1, max_value=5),
            ],
        )
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)
        df = gen.generate(stats)

        # With range [1,5] and 10000 samples, all 5 values should appear
        unique_vals = sorted(df["rating"].unique())
        assert unique_vals == [1, 2, 3, 4, 5]

    def test_timestamp_covers_full_range(self):
        """Verify timestamps span the full specified time range."""
        stats = TableStats(
            database_name="db", schema_name="s", table_name="t", num_records=50000,
            columns=[
                ColumnStats(name="logged_at", data_type="timestamp", nullable=False, distinct_count=50000, min_value="2023-01-01 00:00:00", max_value="2023-12-31 23:59:59"),
            ],
        )
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)
        df = gen.generate(stats)

        timestamps = pd.to_datetime(df["logged_at"])
        # Should cover all 12 months of 2023
        months = timestamps.dt.month.unique()
        assert len(months) == 12
