"""Tests for foreign key relationships and constraint enforcement."""

import sqlite3

import numpy as np
import pandas as pd
import pytest

from numbers_dont_lie.generator import SyntheticDataGenerator
from numbers_dont_lie.models import ColumnStats, Constraint, ForeignKey, TableStats
from numbers_dont_lie.orchestrator import Orchestrator
from numbers_dont_lie.readers.sqlite import SQLiteStatsReader, SQLiteWriter


# --- FK Relationship Tests ---


class TestForeignKeyOneToOne:
    """Test one-to-one FK relationships (child distinct_count == parent row count)."""

    def test_one_to_one_all_parent_values_used(self):
        """In 1:1, child FK values should all come from parent PK."""
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)

        # Generate parent table first
        parent_stats = TableStats(
            database_name="db", schema_name="public", table_name="users",
            num_records=100,
            columns=[
                ColumnStats(name="user_id", data_type="integer", nullable=False, distinct_count=100, min_value=1, max_value=100, is_unique=True),
                ColumnStats(name="name", data_type="varchar", nullable=False, distinct_count=100, avg_length=10.0),
            ],
            constraints=[Constraint(name="pk_users", constraint_type="PRIMARY KEY", columns=["user_id"])],
        )
        parent_df = gen.generate(parent_stats)

        # Generate child with FK referencing parent
        child_stats = TableStats(
            database_name="db", schema_name="public", table_name="profiles",
            num_records=100,
            columns=[
                ColumnStats(name="profile_id", data_type="integer", nullable=False, distinct_count=100, min_value=1, max_value=100, is_unique=True),
                ColumnStats(name="user_id", data_type="integer", nullable=False, distinct_count=100, min_value=1, max_value=100),
                ColumnStats(name="bio", data_type="varchar", nullable=True, distinct_count=80, null_count=20, avg_length=50.0),
            ],
            foreign_keys=[
                ForeignKey(
                    child_schema="public", child_table="profiles", child_columns=["user_id"],
                    parent_schema="public", parent_table="users", parent_columns=["user_id"],
                ),
            ],
        )
        child_df = gen.generate(child_stats)

        # All child FK values must exist in parent
        parent_ids = set(parent_df["user_id"].dropna())
        child_fk_values = set(child_df["user_id"].dropna())
        assert child_fk_values.issubset(parent_ids)

    def test_one_to_one_no_orphans(self):
        """Every FK value in child should reference a valid parent."""
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=99)

        parent_stats = TableStats(
            database_name="db", schema_name="dbo", table_name="employees",
            num_records=50,
            columns=[
                ColumnStats(name="emp_id", data_type="integer", nullable=False, distinct_count=50, min_value=1000, max_value=1049, is_unique=True),
            ],
            constraints=[Constraint(name="pk_emp", constraint_type="PRIMARY KEY", columns=["emp_id"])],
        )
        parent_df = gen.generate(parent_stats)

        child_stats = TableStats(
            database_name="db", schema_name="dbo", table_name="badges",
            num_records=50,
            columns=[
                ColumnStats(name="badge_id", data_type="integer", nullable=False, distinct_count=50, min_value=1, max_value=50),
                ColumnStats(name="emp_id", data_type="integer", nullable=False, distinct_count=50, min_value=1000, max_value=1049),
            ],
            foreign_keys=[
                ForeignKey(
                    child_schema="dbo", child_table="badges", child_columns=["emp_id"],
                    parent_schema="dbo", parent_table="employees", parent_columns=["emp_id"],
                ),
            ],
        )
        child_df = gen.generate(child_stats)

        # No orphan FK values
        valid_ids = set(parent_df["emp_id"].values)
        child_ids = set(child_df["emp_id"].values)
        orphans = child_ids - valid_ids
        assert len(orphans) == 0, f"Found orphan FK values: {orphans}"


class TestForeignKeyOneToManyLow:
    """Test one-to-many FK with low cardinality (few parents, many children each)."""

    def test_low_cardinality_fk_few_parents(self):
        """5 parent rows, 1000 child rows - each parent referenced ~200 times."""
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)

        parent_stats = TableStats(
            database_name="db", schema_name="public", table_name="categories",
            num_records=5,
            columns=[
                ColumnStats(name="cat_id", data_type="integer", nullable=False, distinct_count=5, min_value=1, max_value=5, is_unique=True),
                ColumnStats(name="cat_name", data_type="varchar", nullable=False, distinct_count=5, avg_length=10.0),
            ],
            constraints=[Constraint(name="pk_cat", constraint_type="PRIMARY KEY", columns=["cat_id"])],
        )
        parent_df = gen.generate(parent_stats)

        child_stats = TableStats(
            database_name="db", schema_name="public", table_name="products",
            num_records=1000,
            columns=[
                ColumnStats(name="product_id", data_type="integer", nullable=False, distinct_count=1000, min_value=1, max_value=1000, is_unique=True),
                ColumnStats(name="cat_id", data_type="integer", nullable=False, distinct_count=5, min_value=1, max_value=5),
                ColumnStats(name="product_name", data_type="varchar", nullable=False, distinct_count=1000, avg_length=20.0),
            ],
            foreign_keys=[
                ForeignKey(
                    child_schema="public", child_table="products", child_columns=["cat_id"],
                    parent_schema="public", parent_table="categories", parent_columns=["cat_id"],
                ),
            ],
        )
        child_df = gen.generate(child_stats)

        # All child FK values come from parent
        valid_ids = set(parent_df["cat_id"].values)
        child_fk_values = set(child_df["cat_id"].values)
        assert child_fk_values.issubset(valid_ids)

        # With uniform distribution, each parent should be referenced roughly equally
        counts = child_df["cat_id"].value_counts()
        expected = 1000 / 5  # 200 each
        for cat_id, count in counts.items():
            assert count > expected * 0.5, f"Category {cat_id} underrepresented: {count}"
            assert count < expected * 1.5, f"Category {cat_id} overrepresented: {count}"

    def test_low_cardinality_fk_nullable(self):
        """FK column with nulls - some children have no parent reference."""
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)

        parent_stats = TableStats(
            database_name="db", schema_name="public", table_name="departments",
            num_records=3,
            columns=[
                ColumnStats(name="dept_id", data_type="integer", nullable=False, distinct_count=3, min_value=1, max_value=3, is_unique=True),
            ],
            constraints=[Constraint(name="pk_dept", constraint_type="PRIMARY KEY", columns=["dept_id"])],
        )
        parent_df = gen.generate(parent_stats)

        child_stats = TableStats(
            database_name="db", schema_name="public", table_name="staff",
            num_records=500,
            columns=[
                ColumnStats(name="staff_id", data_type="integer", nullable=False, distinct_count=500, min_value=1, max_value=500),
                ColumnStats(name="dept_id", data_type="integer", nullable=True, distinct_count=3, null_count=100, min_value=1, max_value=3),
            ],
            foreign_keys=[
                ForeignKey(
                    child_schema="public", child_table="staff", child_columns=["dept_id"],
                    parent_schema="public", parent_table="departments", parent_columns=["dept_id"],
                ),
            ],
        )
        child_df = gen.generate(child_stats)

        # Non-null FK values should all reference parent
        non_null_fk = child_df["dept_id"].dropna()
        valid_ids = set(parent_df["dept_id"].values)
        assert set(non_null_fk.unique()).issubset(valid_ids)

        # Should have some nulls
        assert child_df["dept_id"].isna().sum() > 0


class TestForeignKeyOneToManyMedium:
    """Test one-to-many FK with medium cardinality."""

    def test_medium_cardinality_fk(self):
        """100 parents, 10000 children - each parent referenced ~100 times."""
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)

        parent_stats = TableStats(
            database_name="db", schema_name="sales", table_name="customers",
            num_records=100,
            columns=[
                ColumnStats(name="customer_id", data_type="integer", nullable=False, distinct_count=100, min_value=1, max_value=100, is_unique=True),
                ColumnStats(name="name", data_type="varchar", nullable=False, distinct_count=100, avg_length=15.0),
            ],
            constraints=[Constraint(name="pk_cust", constraint_type="PRIMARY KEY", columns=["customer_id"])],
        )
        parent_df = gen.generate(parent_stats)

        child_stats = TableStats(
            database_name="db", schema_name="sales", table_name="orders",
            num_records=10000,
            columns=[
                ColumnStats(name="order_id", data_type="integer", nullable=False, distinct_count=10000, min_value=1, max_value=10000, is_unique=True),
                ColumnStats(name="customer_id", data_type="integer", nullable=False, distinct_count=100, min_value=1, max_value=100),
                ColumnStats(name="total", data_type="decimal", nullable=False, distinct_count=5000, min_value=1.0, max_value=9999.99),
            ],
            foreign_keys=[
                ForeignKey(
                    child_schema="sales", child_table="orders", child_columns=["customer_id"],
                    parent_schema="sales", parent_table="customers", parent_columns=["customer_id"],
                ),
            ],
        )
        child_df = gen.generate(child_stats)

        # Referential integrity
        valid_ids = set(parent_df["customer_id"].values)
        child_fk_values = set(child_df["customer_id"].values)
        assert child_fk_values.issubset(valid_ids)

        # Cardinality preserved: child should reference ~100 distinct parents
        assert child_df["customer_id"].nunique() <= 100

    def test_medium_cardinality_distribution_spread(self):
        """All parents should be referenced at least once with enough children."""
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)

        parent_stats = TableStats(
            database_name="db", schema_name="s", table_name="parent",
            num_records=50,
            columns=[
                ColumnStats(name="id", data_type="integer", nullable=False, distinct_count=50, min_value=1, max_value=50, is_unique=True),
            ],
            constraints=[Constraint(name="pk", constraint_type="PRIMARY KEY", columns=["id"])],
        )
        parent_df = gen.generate(parent_stats)

        child_stats = TableStats(
            database_name="db", schema_name="s", table_name="child",
            num_records=5000,
            columns=[
                ColumnStats(name="child_id", data_type="integer", nullable=False, distinct_count=5000, min_value=1, max_value=5000),
                ColumnStats(name="parent_id", data_type="integer", nullable=False, distinct_count=50, min_value=1, max_value=50),
            ],
            foreign_keys=[
                ForeignKey(
                    child_schema="s", child_table="child", child_columns=["parent_id"],
                    parent_schema="s", parent_table="parent", parent_columns=["id"],
                ),
            ],
        )
        child_df = gen.generate(child_stats)

        # With 5000 children and 50 parents, uniform sampling should hit all parents
        referenced_parents = child_df["parent_id"].nunique()
        assert referenced_parents == len(parent_df), f"Only {referenced_parents}/50 parents referenced"


class TestForeignKeyOneToManyHigh:
    """Test one-to-many FK with high cardinality (many parents, few children each)."""

    def test_high_cardinality_fk(self):
        """10000 parents, 15000 children - most parents referenced 1-2 times."""
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)

        parent_stats = TableStats(
            database_name="db", schema_name="public", table_name="accounts",
            num_records=10000,
            columns=[
                ColumnStats(name="account_id", data_type="integer", nullable=False, distinct_count=10000, min_value=1, max_value=10000, is_unique=True),
            ],
            constraints=[Constraint(name="pk_acc", constraint_type="PRIMARY KEY", columns=["account_id"])],
        )
        parent_df = gen.generate(parent_stats)

        child_stats = TableStats(
            database_name="db", schema_name="public", table_name="transactions",
            num_records=15000,
            columns=[
                ColumnStats(name="txn_id", data_type="integer", nullable=False, distinct_count=15000, min_value=1, max_value=15000, is_unique=True),
                ColumnStats(name="account_id", data_type="integer", nullable=False, distinct_count=10000, min_value=1, max_value=10000),
                ColumnStats(name="amount", data_type="decimal", nullable=False, distinct_count=10000, min_value=-5000.0, max_value=50000.0),
            ],
            foreign_keys=[
                ForeignKey(
                    child_schema="public", child_table="transactions", child_columns=["account_id"],
                    parent_schema="public", parent_table="accounts", parent_columns=["account_id"],
                ),
            ],
        )
        child_df = gen.generate(child_stats)

        # Referential integrity
        valid_ids = set(parent_df["account_id"].values)
        child_fk_values = set(child_df["account_id"].values)
        assert child_fk_values.issubset(valid_ids)

        # High cardinality: with 15000 samples from 10000 uniform, expect ~7800+ unique
        # (1 - (9999/10000)^15000 = ~0.78 coverage per value)
        referenced = child_df["account_id"].nunique()
        assert referenced > 7500, f"Only {referenced}/10000 parents referenced"

    def test_high_cardinality_average_references(self):
        """Average references per parent should be close to child_rows/parent_rows."""
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)

        parent_stats = TableStats(
            database_name="db", schema_name="s", table_name="parent_big",
            num_records=5000,
            columns=[
                ColumnStats(name="id", data_type="integer", nullable=False, distinct_count=5000, min_value=1, max_value=5000, is_unique=True),
            ],
            constraints=[Constraint(name="pk", constraint_type="PRIMARY KEY", columns=["id"])],
        )
        parent_df = gen.generate(parent_stats)

        child_stats = TableStats(
            database_name="db", schema_name="s", table_name="child_big",
            num_records=20000,
            columns=[
                ColumnStats(name="child_id", data_type="integer", nullable=False, distinct_count=20000, min_value=1, max_value=20000),
                ColumnStats(name="parent_id", data_type="integer", nullable=False, distinct_count=5000, min_value=1, max_value=5000),
            ],
            foreign_keys=[
                ForeignKey(
                    child_schema="s", child_table="child_big", child_columns=["parent_id"],
                    parent_schema="s", parent_table="parent_big", parent_columns=["id"],
                ),
            ],
        )
        child_df = gen.generate(child_stats)

        # Expected avg references: 20000/5000 = 4
        counts = child_df["parent_id"].value_counts()
        avg_refs = counts.mean()
        assert 3.0 < avg_refs < 5.0, f"Average references: {avg_refs}"


class TestForeignKeyMultiColumn:
    """Test composite (multi-column) foreign keys."""

    def test_composite_fk_integrity(self):
        """Composite FK (2 columns) should reference valid parent combinations."""
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)

        parent_stats = TableStats(
            database_name="db", schema_name="public", table_name="price_list",
            num_records=50,
            columns=[
                ColumnStats(name="product_id", data_type="integer", nullable=False, distinct_count=10, min_value=1, max_value=10),
                ColumnStats(name="region_id", data_type="integer", nullable=False, distinct_count=5, min_value=1, max_value=5),
                ColumnStats(name="price", data_type="decimal", nullable=False, distinct_count=50, min_value=1.0, max_value=100.0),
            ],
            constraints=[Constraint(name="pk_price", constraint_type="PRIMARY KEY", columns=["product_id", "region_id"])],
        )
        parent_df = gen.generate(parent_stats)

        child_stats = TableStats(
            database_name="db", schema_name="public", table_name="sales",
            num_records=500,
            columns=[
                ColumnStats(name="sale_id", data_type="integer", nullable=False, distinct_count=500, min_value=1, max_value=500),
                ColumnStats(name="product_id", data_type="integer", nullable=False, distinct_count=10, min_value=1, max_value=10),
                ColumnStats(name="region_id", data_type="integer", nullable=False, distinct_count=5, min_value=1, max_value=5),
                ColumnStats(name="quantity", data_type="integer", nullable=False, distinct_count=20, min_value=1, max_value=20),
            ],
            foreign_keys=[
                ForeignKey(
                    child_schema="public", child_table="sales",
                    child_columns=["product_id", "region_id"],
                    parent_schema="public", parent_table="price_list",
                    parent_columns=["product_id", "region_id"],
                ),
            ],
        )
        child_df = gen.generate(child_stats)

        # Every (product_id, region_id) in child must exist in parent
        parent_pairs = set(zip(parent_df["product_id"], parent_df["region_id"]))
        child_pairs = set(zip(child_df["product_id"], child_df["region_id"]))
        orphans = child_pairs - parent_pairs
        assert len(orphans) == 0, f"Orphan composite FK pairs: {orphans}"


# --- Constraint/Unique Index Tests ---


class TestUniqueConstraint:
    """Test that UNIQUE and PRIMARY KEY constraints produce no duplicates."""

    def test_primary_key_no_duplicates(self):
        """PK column should have all unique values."""
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)

        stats = TableStats(
            database_name="db", schema_name="s", table_name="t", num_records=1000,
            columns=[
                ColumnStats(name="id", data_type="integer", nullable=False, distinct_count=1000, min_value=1, max_value=1000000, is_unique=True),
                ColumnStats(name="value", data_type="varchar", nullable=False, distinct_count=500, avg_length=10.0),
            ],
            constraints=[Constraint(name="pk_t", constraint_type="PRIMARY KEY", columns=["id"])],
        )
        df = gen.generate(stats)
        assert df["id"].nunique() == 1000, f"Got {df['id'].nunique()} unique values, expected 1000"

    def test_unique_index_no_duplicates(self):
        """Column with UNIQUE constraint should have no duplicates."""
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)

        stats = TableStats(
            database_name="db", schema_name="s", table_name="t", num_records=500,
            columns=[
                ColumnStats(name="id", data_type="integer", nullable=False, distinct_count=500, min_value=1, max_value=500, is_unique=True),
                ColumnStats(name="email", data_type="varchar", nullable=False, distinct_count=500, avg_length=25.0, is_unique=True),
                ColumnStats(name="name", data_type="varchar", nullable=False, distinct_count=400, avg_length=12.0),
            ],
            constraints=[
                Constraint(name="pk", constraint_type="PRIMARY KEY", columns=["id"]),
                Constraint(name="uq_email", constraint_type="UNIQUE", columns=["email"]),
            ],
        )
        df = gen.generate(stats)
        assert df["email"].nunique() == 500

    def test_composite_unique_no_duplicates(self):
        """Composite UNIQUE constraint should have no duplicate combinations."""
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)

        stats = TableStats(
            database_name="db", schema_name="s", table_name="enrollments", num_records=1000,
            columns=[
                ColumnStats(name="student_id", data_type="integer", nullable=False, distinct_count=200, min_value=1, max_value=200),
                ColumnStats(name="course_id", data_type="integer", nullable=False, distinct_count=50, min_value=1, max_value=50),
                ColumnStats(name="grade", data_type="varchar", nullable=True, distinct_count=5, null_count=100, avg_length=2.0),
            ],
            constraints=[
                Constraint(name="uq_enrollment", constraint_type="UNIQUE", columns=["student_id", "course_id"]),
            ],
        )
        df = gen.generate(stats)
        duplicates = df.duplicated(subset=["student_id", "course_id"]).sum()
        assert duplicates == 0, f"Found {duplicates} duplicate (student_id, course_id) pairs"

    def test_unique_with_large_dataset(self):
        """Unique constraint holds even with large datasets."""
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)

        stats = TableStats(
            database_name="db", schema_name="s", table_name="t", num_records=10000,
            columns=[
                ColumnStats(name="code", data_type="varchar", nullable=False, distinct_count=10000, avg_length=8.0, is_unique=True),
            ],
            constraints=[Constraint(name="uq_code", constraint_type="UNIQUE", columns=["code"])],
        )
        df = gen.generate(stats)
        assert df["code"].nunique() == 10000


class TestUniqueWithFK:
    """Test that unique constraints and FK references work together."""

    def test_unique_fk_column(self):
        """1:1 relationship where FK column is also unique (e.g., user_profile.user_id UNIQUE FK)."""
        gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)

        parent_stats = TableStats(
            database_name="db", schema_name="s", table_name="users",
            num_records=100,
            columns=[
                ColumnStats(name="id", data_type="integer", nullable=False, distinct_count=100, min_value=1, max_value=100, is_unique=True),
            ],
            constraints=[Constraint(name="pk", constraint_type="PRIMARY KEY", columns=["id"])],
        )
        parent_df = gen.generate(parent_stats)

        child_stats = TableStats(
            database_name="db", schema_name="s", table_name="profiles",
            num_records=100,
            columns=[
                ColumnStats(name="profile_id", data_type="integer", nullable=False, distinct_count=100, min_value=1, max_value=100, is_unique=True),
                ColumnStats(name="user_id", data_type="integer", nullable=False, distinct_count=100, min_value=1, max_value=100, is_unique=True),
                ColumnStats(name="bio", data_type="varchar", nullable=True, distinct_count=80, null_count=20, avg_length=50.0),
            ],
            foreign_keys=[
                ForeignKey(
                    child_schema="s", child_table="profiles", child_columns=["user_id"],
                    parent_schema="s", parent_table="users", parent_columns=["id"],
                ),
            ],
            constraints=[
                Constraint(name="pk_prof", constraint_type="PRIMARY KEY", columns=["profile_id"]),
                Constraint(name="uq_user", constraint_type="UNIQUE", columns=["user_id"]),
            ],
        )
        child_df = gen.generate(child_stats)

        # FK integrity
        valid_ids = set(parent_df["id"].values)
        child_fk_values = set(child_df["user_id"].values)
        assert child_fk_values.issubset(valid_ids)

        # Uniqueness
        assert child_df["user_id"].nunique() == 100


# --- SQLite Integration Tests for FK Discovery ---


class TestSQLiteFKDiscovery:
    """Test that the SQLite reader correctly discovers FKs and constraints."""

    @pytest.fixture
    def fk_db(self, tmp_path):
        """Create a database with FK relationships."""
        db_path = str(tmp_path / "fk_test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")

        conn.execute("""
            CREATE TABLE departments (
                dept_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE
            )
        """)
        conn.execute("""
            CREATE TABLE employees (
                emp_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                dept_id INTEGER NOT NULL,
                FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
            )
        """)
        conn.execute("""
            CREATE TABLE projects (
                project_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                lead_emp_id INTEGER,
                FOREIGN KEY (lead_emp_id) REFERENCES employees(emp_id)
            )
        """)

        # Insert data
        for i in range(1, 6):
            conn.execute("INSERT INTO departments VALUES (?, ?)", (i, f"Dept_{i}"))
        for i in range(1, 51):
            conn.execute("INSERT INTO employees VALUES (?, ?, ?, ?)", (i, f"Emp_{i}", f"emp{i}@co.com", (i % 5) + 1))
        for i in range(1, 21):
            conn.execute("INSERT INTO projects VALUES (?, ?, ?)", (i, f"Project_{i}", (i % 50) + 1))

        conn.commit()
        conn.close()
        return db_path

    def test_discovers_foreign_keys(self, fk_db):
        with SQLiteStatsReader(fk_db) as reader:
            stats = reader.read_table_stats("main", "employees")
            assert len(stats.foreign_keys) == 1
            fk = stats.foreign_keys[0]
            assert fk.parent_table == "departments"
            assert fk.child_columns == ["dept_id"]
            assert fk.parent_columns == ["dept_id"]

    def test_discovers_primary_key(self, fk_db):
        with SQLiteStatsReader(fk_db) as reader:
            stats = reader.read_table_stats("main", "employees")
            pk_constraints = [c for c in stats.constraints if c.constraint_type == "PRIMARY KEY"]
            assert len(pk_constraints) == 1
            assert pk_constraints[0].columns == ["emp_id"]

    def test_discovers_unique_constraint(self, fk_db):
        with SQLiteStatsReader(fk_db) as reader:
            stats = reader.read_table_stats("main", "employees")
            uq_constraints = [c for c in stats.constraints if c.constraint_type == "UNIQUE"]
            # Should find the UNIQUE index on email
            email_uq = [c for c in uq_constraints if "email" in c.columns]
            assert len(email_uq) >= 1

    def test_marks_unique_columns(self, fk_db):
        with SQLiteStatsReader(fk_db) as reader:
            stats = reader.read_table_stats("main", "employees")
            email_col = stats.get_column("email")
            assert email_col.is_unique is True

    def test_full_pipeline_with_fk(self, fk_db, tmp_path):
        """End-to-end: generate synthetic data preserving FK relationships."""
        dest_db = str(tmp_path / "synthetic_fk.db")

        with SQLiteStatsReader(fk_db) as reader, SQLiteWriter(dest_db) as writer:
            # Read stats for all tables
            dept_stats = reader.read_table_stats("main", "departments")
            emp_stats = reader.read_table_stats("main", "employees")

            # Generate in dependency order
            gen = SyntheticDataGenerator(scale_factor=1.0, seed=42)

            dept_df = gen.generate(dept_stats)
            writer.write_table(dept_df, "main", "departments")

            emp_df = gen.generate(emp_stats)
            writer.write_table(emp_df, "main", "employees")

        # Verify FK integrity in synthetic data
        conn = sqlite3.connect(dest_db)
        dept_ids = set(r[0] for r in conn.execute("SELECT dept_id FROM departments").fetchall())
        emp_dept_ids = set(r[0] for r in conn.execute("SELECT dept_id FROM employees").fetchall())
        conn.close()

        orphans = emp_dept_ids - dept_ids
        assert len(orphans) == 0, f"Orphan dept_id values in employees: {orphans}"
