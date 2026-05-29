"""Integration tests for SQLite reader and writer."""

import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from numbers_dont_lie.generator import SyntheticDataGenerator
from numbers_dont_lie.orchestrator import Orchestrator
from numbers_dont_lie.readers.sqlite import SQLiteStatsReader, SQLiteWriter


@pytest.fixture
def sample_db(tmp_path):
    """Create a temporary SQLite database with sample data."""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT,
            age INTEGER,
            balance REAL,
            is_active INTEGER,
            created_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            product TEXT NOT NULL,
            quantity INTEGER,
            price REAL,
            order_date TEXT
        )
    """)

    # Insert sample data into customers
    customers = [
        (i, f"User_{i}", f"user{i}@example.com" if i % 10 != 0 else None, 20 + (i % 60), 1000.0 + i * 10, i % 2, f"2023-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}")
        for i in range(1, 501)
    ]
    cur.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?, ?)", customers)

    # Insert sample data into orders
    orders = [
        (i, (i % 500) + 1, f"Product_{i % 20}", 1 + (i % 10), 9.99 + (i % 100), f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}")
        for i in range(1, 2001)
    ]
    cur.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)", orders)

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def dest_db(tmp_path):
    """Path for destination database."""
    return str(tmp_path / "synthetic.db")


class TestSQLiteStatsReader:
    def test_connect_disconnect(self, sample_db):
        reader = SQLiteStatsReader(sample_db)
        reader.connect()
        reader.disconnect()

    def test_connect_nonexistent_raises(self, tmp_path):
        reader = SQLiteStatsReader(str(tmp_path / "nonexistent.db"))
        with pytest.raises(FileNotFoundError):
            reader.connect()

    def test_context_manager(self, sample_db):
        with SQLiteStatsReader(sample_db) as reader:
            tables = reader.list_tables()
            assert len(tables) > 0

    def test_list_tables(self, sample_db):
        with SQLiteStatsReader(sample_db) as reader:
            tables = reader.list_tables()
            table_names = [t[2] for t in tables]
            assert "customers" in table_names
            assert "orders" in table_names

    def test_read_table_stats_row_count(self, sample_db):
        with SQLiteStatsReader(sample_db) as reader:
            stats = reader.read_table_stats("main", "customers")
            assert stats.num_records == 500

    def test_read_table_stats_columns(self, sample_db):
        with SQLiteStatsReader(sample_db) as reader:
            stats = reader.read_table_stats("main", "customers")
            col_names = [c.name for c in stats.columns]
            assert "id" in col_names
            assert "name" in col_names
            assert "email" in col_names
            assert len(stats.columns) == 7

    def test_read_table_stats_distinct_count(self, sample_db):
        with SQLiteStatsReader(sample_db) as reader:
            stats = reader.read_table_stats("main", "customers")
            id_col = stats.get_column("id")
            assert id_col.distinct_count == 500

    def test_read_table_stats_null_count(self, sample_db):
        with SQLiteStatsReader(sample_db) as reader:
            stats = reader.read_table_stats("main", "customers")
            email_col = stats.get_column("email")
            # Every 10th customer has NULL email (50 out of 500)
            assert email_col.null_count == 50

    def test_read_table_stats_min_max(self, sample_db):
        with SQLiteStatsReader(sample_db) as reader:
            stats = reader.read_table_stats("main", "customers")
            id_col = stats.get_column("id")
            assert id_col.min_value == 1
            assert id_col.max_value == 500


class TestSQLiteWriter:
    def test_write_creates_table(self, dest_db):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        with SQLiteWriter(dest_db) as writer:
            rows = writer.write_table(df, "main", "test_table")

        assert rows == 3
        # Verify data was written
        conn = sqlite3.connect(dest_db)
        result = conn.execute("SELECT COUNT(*) FROM test_table").fetchone()[0]
        conn.close()
        assert result == 3

    def test_write_replace_overwrites(self, dest_db):
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5]})

        with SQLiteWriter(dest_db) as writer:
            writer.write_table(df1, "main", "test_table")
            writer.write_table(df2, "main", "test_table", if_exists="replace")

        conn = sqlite3.connect(dest_db)
        result = conn.execute("SELECT COUNT(*) FROM test_table").fetchone()[0]
        conn.close()
        assert result == 2


class TestSQLiteEndToEnd:
    def test_full_pipeline_single_table(self, sample_db, dest_db):
        with SQLiteStatsReader(sample_db) as reader, SQLiteWriter(dest_db) as writer:
            orchestrator = Orchestrator(reader=reader, writer=writer, scale_factor=1.0, seed=42)
            results = orchestrator.run(schema="main", tables=["customers"], dest_schema="main")

        assert len(results) == 1
        assert results[0].status == "SUCCESS"
        assert results[0].generated_rows == 500

        # Verify destination has data
        conn = sqlite3.connect(dest_db)
        count = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        conn.close()
        assert count == 500

    def test_full_pipeline_all_tables(self, sample_db, dest_db):
        with SQLiteStatsReader(sample_db) as reader, SQLiteWriter(dest_db) as writer:
            orchestrator = Orchestrator(reader=reader, writer=writer, scale_factor=1.0, seed=42)
            results = orchestrator.run(dest_schema="main")

        assert len(results) == 2
        assert all(r.status == "SUCCESS" for r in results)

    def test_scale_factor_applied(self, sample_db, dest_db):
        with SQLiteStatsReader(sample_db) as reader, SQLiteWriter(dest_db) as writer:
            orchestrator = Orchestrator(reader=reader, writer=writer, scale_factor=2.0, seed=42)
            results = orchestrator.run(schema="main", tables=["customers"], dest_schema="main")

        assert results[0].generated_rows == 1000

        conn = sqlite3.connect(dest_db)
        count = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        conn.close()
        assert count == 1000

    def test_generated_data_has_correct_columns(self, sample_db, dest_db):
        with SQLiteStatsReader(sample_db) as reader, SQLiteWriter(dest_db) as writer:
            orchestrator = Orchestrator(reader=reader, writer=writer, scale_factor=1.0, seed=42)
            orchestrator.run(schema="main", tables=["customers"], dest_schema="main")

        conn = sqlite3.connect(dest_db)
        cur = conn.execute("PRAGMA table_info(customers)")
        col_names = [row[1] for row in cur.fetchall()]
        conn.close()

        assert "id" in col_names
        assert "name" in col_names
        assert "email" in col_names
        assert "age" in col_names
        assert "balance" in col_names
