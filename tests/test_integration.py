"""Integration tests for PostgreSQL, MySQL, and SQL Server (ODBC).

These tests require running database instances. In GitHub Actions, they are
provided via service containers. Locally, they are skipped unless the
corresponding environment variables are set.

Environment variables:
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
    MYSQL_HOST, MYSQL_PORT, MYSQL_DB, MYSQL_USER, MYSQL_PASSWORD
    MSSQL_CONNECTION_STRING (full ODBC connection string)
"""

import os

import pandas as pd
import pytest

from numbers_dont_lie.generator import SyntheticDataGenerator
from numbers_dont_lie.models import TableStats
from numbers_dont_lie.orchestrator import Orchestrator


# --- PostgreSQL ---

def _pg_config():
    return {
        "host": os.environ.get("POSTGRES_HOST", "localhost"),
        "port": int(os.environ.get("POSTGRES_PORT", "5432")),
        "database": os.environ.get("POSTGRES_DB", "testdb"),
        "user": os.environ.get("POSTGRES_USER", "postgres"),
        "password": os.environ.get("POSTGRES_PASSWORD", "postgres"),
    }


def _pg_available():
    try:
        import psycopg2
        cfg = _pg_config()
        conn = psycopg2.connect(
            host=cfg["host"], port=cfg["port"],
            dbname=cfg["database"], user=cfg["user"], password=cfg["password"],
        )
        conn.close()
        return True
    except Exception:
        return False


requires_postgres = pytest.mark.skipif(
    not _pg_available(),
    reason="PostgreSQL not available",
)


@pytest.fixture
def pg_setup():
    """Set up test tables in PostgreSQL."""
    import psycopg2

    cfg = _pg_config()
    conn = psycopg2.connect(
        host=cfg["host"], port=cfg["port"],
        dbname=cfg["database"], user=cfg["user"], password=cfg["password"],
    )
    conn.autocommit = True
    cur = conn.cursor()

    # Create test schema and table
    cur.execute("DROP SCHEMA IF EXISTS ndl_test CASCADE")
    cur.execute("CREATE SCHEMA ndl_test")
    cur.execute("""
        CREATE TABLE ndl_test.sample (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            value DOUBLE PRECISION,
            category VARCHAR(20),
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)

    # Insert test data
    for i in range(200):
        cur.execute(
            "INSERT INTO ndl_test.sample (name, value, category, created_at) VALUES (%s, %s, %s, NOW() - INTERVAL '%s days')",
            (f"item_{i}", float(i) * 1.5, f"cat_{i % 5}", i),
        )

    # Run ANALYZE so pg_stats is populated
    cur.execute("ANALYZE ndl_test.sample")

    cur.close()
    conn.close()

    yield cfg

    # Cleanup
    conn = psycopg2.connect(
        host=cfg["host"], port=cfg["port"],
        dbname=cfg["database"], user=cfg["user"], password=cfg["password"],
    )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("DROP SCHEMA IF EXISTS ndl_test CASCADE")
    cur.execute("DROP SCHEMA IF EXISTS ndl_synth CASCADE")
    cur.close()
    conn.close()


@requires_postgres
class TestPostgres:
    def test_list_tables(self, pg_setup):
        from numbers_dont_lie.readers.postgres import PostgresStatsReader

        cfg = pg_setup
        with PostgresStatsReader(**cfg) as reader:
            tables = reader.list_tables(schema="ndl_test")
            table_names = [t[2] for t in tables]
            assert "sample" in table_names

    def test_read_stats(self, pg_setup):
        from numbers_dont_lie.readers.postgres import PostgresStatsReader

        cfg = pg_setup
        with PostgresStatsReader(**cfg) as reader:
            stats = reader.read_table_stats("ndl_test", "sample")
            assert stats.num_records >= 190  # pg_class estimate may be approximate
            assert len(stats.columns) == 5
            assert stats.get_column("name") is not None

    def test_full_pipeline(self, pg_setup):
        from numbers_dont_lie.readers.postgres import PostgresStatsReader, PostgresWriter
        import psycopg2

        cfg = pg_setup

        # Create destination schema
        conn = psycopg2.connect(
            host=cfg["host"], port=cfg["port"],
            dbname=cfg["database"], user=cfg["user"], password=cfg["password"],
        )
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("CREATE SCHEMA IF NOT EXISTS ndl_synth")
        cur.close()
        conn.close()

        with PostgresStatsReader(**cfg) as reader, PostgresWriter(**cfg) as writer:
            orchestrator = Orchestrator(reader=reader, writer=writer, scale_factor=1.0, seed=42)
            results = orchestrator.run(schema="ndl_test", dest_schema="ndl_synth")

        assert len(results) == 1
        assert results[0].status == "SUCCESS"
        assert results[0].generated_rows > 0


# --- MySQL ---

def _mysql_config():
    return {
        "host": os.environ.get("MYSQL_HOST", "localhost"),
        "port": int(os.environ.get("MYSQL_PORT", "3306")),
        "database": os.environ.get("MYSQL_DB", "testdb"),
        "user": os.environ.get("MYSQL_USER", "root"),
        "password": os.environ.get("MYSQL_PASSWORD", "root"),
    }


def _mysql_available():
    try:
        import mysql.connector
        cfg = _mysql_config()
        conn = mysql.connector.connect(
            host=cfg["host"], port=cfg["port"],
            database=cfg["database"], user=cfg["user"], password=cfg["password"],
        )
        conn.close()
        return True
    except Exception:
        return False


requires_mysql = pytest.mark.skipif(
    not _mysql_available(),
    reason="MySQL not available",
)


@pytest.fixture
def mysql_setup():
    """Set up test tables in MySQL."""
    import mysql.connector

    cfg = _mysql_config()
    conn = mysql.connector.connect(
        host=cfg["host"], port=cfg["port"],
        database=cfg["database"], user=cfg["user"], password=cfg["password"],
    )
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS ndl_sample")
    cur.execute("""
        CREATE TABLE ndl_sample (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            value DOUBLE,
            category VARCHAR(20),
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    for i in range(200):
        cur.execute(
            "INSERT INTO ndl_sample (name, value, category) VALUES (%s, %s, %s)",
            (f"item_{i}", float(i) * 1.5, f"cat_{i % 5}"),
        )

    conn.commit()

    # Run ANALYZE to populate statistics
    cur.execute("ANALYZE TABLE ndl_sample")

    cur.close()
    conn.close()

    yield cfg

    # Cleanup
    conn = mysql.connector.connect(
        host=cfg["host"], port=cfg["port"],
        database=cfg["database"], user=cfg["user"], password=cfg["password"],
    )
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS ndl_sample")
    cur.execute("DROP TABLE IF EXISTS ndl_sample_synth")
    conn.commit()
    cur.close()
    conn.close()


@requires_mysql
class TestMySQL:
    def test_list_tables(self, mysql_setup):
        from numbers_dont_lie.readers.mysql import MySQLStatsReader

        cfg = mysql_setup
        with MySQLStatsReader(**cfg) as reader:
            tables = reader.list_tables()
            table_names = [t[2] for t in tables]
            assert "ndl_sample" in table_names

    def test_read_stats(self, mysql_setup):
        from numbers_dont_lie.readers.mysql import MySQLStatsReader

        cfg = mysql_setup
        with MySQLStatsReader(**cfg) as reader:
            stats = reader.read_table_stats(cfg["database"], "ndl_sample")
            assert stats.num_records >= 190  # InnoDB estimates may vary
            assert len(stats.columns) == 5

    def test_full_pipeline(self, mysql_setup):
        from numbers_dont_lie.readers.mysql import MySQLStatsReader, MySQLWriter

        cfg = mysql_setup
        with MySQLStatsReader(**cfg) as reader, MySQLWriter(**cfg) as writer:
            orchestrator = Orchestrator(reader=reader, writer=writer, scale_factor=1.0, seed=42)
            results = orchestrator.run(schema=cfg["database"], tables=["ndl_sample"], dest_schema=cfg["database"])

        assert len(results) == 1
        assert results[0].status == "SUCCESS"


# --- SQL Server (ODBC) ---

def _mssql_connstr():
    return os.environ.get(
        "MSSQL_CONNECTION_STRING",
        "DRIVER={ODBC Driver 18 for SQL Server};SERVER=localhost,1433;DATABASE=testdb;UID=sa;PWD=YourStrong!Passw0rd;TrustServerCertificate=yes",
    )


def _mssql_available():
    try:
        import pyodbc
        conn = pyodbc.connect(_mssql_connstr(), timeout=5)
        conn.close()
        return True
    except Exception:
        return False


requires_mssql = pytest.mark.skipif(
    not _mssql_available(),
    reason="SQL Server (ODBC) not available",
)


@pytest.fixture
def mssql_setup():
    """Set up test tables in SQL Server."""
    import pyodbc

    connstr = _mssql_connstr()
    conn = pyodbc.connect(connstr)
    conn.autocommit = True
    cur = conn.cursor()

    # Create test schema
    try:
        cur.execute("CREATE SCHEMA ndl_test")
    except Exception:
        pass  # already exists

    try:
        cur.execute("CREATE SCHEMA ndl_synth")
    except Exception:
        pass

    cur.execute("IF OBJECT_ID('ndl_test.sample', 'U') IS NOT NULL DROP TABLE ndl_test.sample")
    cur.execute("""
        CREATE TABLE ndl_test.sample (
            id INT IDENTITY(1,1) PRIMARY KEY,
            name NVARCHAR(100) NOT NULL,
            value FLOAT,
            category NVARCHAR(20),
            created_at DATETIME2 DEFAULT GETDATE()
        )
    """)

    for i in range(200):
        cur.execute(
            "INSERT INTO ndl_test.sample (name, value, category) VALUES (?, ?, ?)",
            (f"item_{i}", float(i) * 1.5, f"cat_{i % 5}"),
        )

    # Update statistics
    cur.execute("UPDATE STATISTICS ndl_test.sample")

    cur.close()
    conn.close()

    yield connstr

    # Cleanup
    conn = pyodbc.connect(connstr)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("IF OBJECT_ID('ndl_test.sample', 'U') IS NOT NULL DROP TABLE ndl_test.sample")
    cur.execute("IF OBJECT_ID('ndl_synth.sample', 'U') IS NOT NULL DROP TABLE ndl_synth.sample")
    cur.close()
    conn.close()


@requires_mssql
class TestMSSQL:
    def test_list_tables(self, mssql_setup):
        from numbers_dont_lie.readers.odbc import ODBCStatsReader

        with ODBCStatsReader(mssql_setup) as reader:
            tables = reader.list_tables(schema="ndl_test")
            table_names = [t[2] for t in tables]
            assert "sample" in table_names

    def test_read_stats(self, mssql_setup):
        from numbers_dont_lie.readers.odbc import ODBCStatsReader

        with ODBCStatsReader(mssql_setup) as reader:
            stats = reader.read_table_stats("ndl_test", "sample")
            assert stats.num_records == 200
            assert len(stats.columns) == 5
            assert stats.get_column("name") is not None

    def test_full_pipeline(self, mssql_setup):
        from numbers_dont_lie.readers.odbc import ODBCStatsReader, ODBCWriter

        with ODBCStatsReader(mssql_setup) as reader, ODBCWriter(mssql_setup) as writer:
            orchestrator = Orchestrator(reader=reader, writer=writer, scale_factor=1.0, seed=42)
            results = orchestrator.run(schema="ndl_test", dest_schema="ndl_synth")

        assert len(results) == 1
        assert results[0].status == "SUCCESS"
        assert results[0].generated_rows == 200
