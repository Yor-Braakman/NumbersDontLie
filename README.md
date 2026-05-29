# Numbers Don't Lie

Privacy-preserving synthetic test data generator for databases.

## Overview

Numbers Don't Lie generates statistically representative synthetic data from database metadata without exposing any real data rows. It supports multiple database backends and provides both a GUI and programmatic API.

## Supported Databases

| Database   | Read Stats | Write Synthetic |
|------------|:----------:|:---------------:|
| PostgreSQL | ✓          | ✓               |
| MySQL      | ✓          | ✓               |
| SQLite     | ✓          | ✓               |
| Delta/Spark| ✓ (notebook) | ✓ (notebook) |

## Installation

### From PyPI (or .whl)

```bash
# Core (SQLite only, no extra dependencies)
pip install numbers-dont-lie

# With PostgreSQL support
pip install numbers-dont-lie[postgres]

# With MySQL support
pip install numbers-dont-lie[mysql]

# With GUI
pip install numbers-dont-lie[gui]

# Everything
pip install numbers-dont-lie[all]
```

### From GitHub Release (.whl)

Download the `.whl` file from [Releases](https://github.com/numbers-dont-lie/numbers-dont-lie/releases) and install:

```bash
pip install numbers_dont_lie-0.1.0-py3-none-any.whl[gui,postgres]
```

## Usage

### GUI Application

```bash
numbers-dont-lie
```

This launches a PySide6 GUI where you can:
1. Configure source database connection
2. Configure destination database connection
3. Set scale factor
4. Generate synthetic data with progress tracking

### Python API

```python
from numbers_dont_lie import SyntheticDataGenerator, Orchestrator
from numbers_dont_lie.readers.postgres import PostgresStatsReader, PostgresWriter

# Connect to source and destination
reader = PostgresStatsReader(host="localhost", port=5432, database="prod_db", user="user", password="pass")
writer = PostgresWriter(host="localhost", port=5432, database="test_db", user="user", password="pass")

reader.connect()
writer.connect()

# Run pipeline
orchestrator = Orchestrator(reader=reader, writer=writer, scale_factor=1.1)
results = orchestrator.run(schema="public", dest_schema="synthetic")

for r in results:
    print(f"{r.table}: {r.status} ({r.generated_rows:,} rows)")

reader.disconnect()
writer.disconnect()
```

### SQLite Example

```python
from numbers_dont_lie.readers.sqlite import SQLiteStatsReader, SQLiteWriter
from numbers_dont_lie import Orchestrator

with SQLiteStatsReader("production.db") as reader, SQLiteWriter("test.db") as writer:
    orchestrator = Orchestrator(reader=reader, writer=writer, scale_factor=1.0)
    results = orchestrator.run(dest_schema="main")
```

### Fabric Notebook (Delta Tables)

For Microsoft Fabric / Delta Lake, use the original notebook: `NumbersDontLie.ipynb`

## Building from Source

```bash
# Install build tools
pip install build

# Build wheel
python -m build

# The .whl file will be in dist/
```

## Privacy Guarantee

The pipeline guarantees:

1. No actual data rows are loaded into memory (except SQLite which uses aggregate queries only)
2. All synthetic data is generated from metadata/statistics only
3. PostgreSQL uses `pg_stats` and `pg_class` system views
4. MySQL uses `information_schema` views
5. SQLite uses aggregate queries (MIN/MAX/COUNT) — no row-level access
6. All generation uses random functions — no deterministic mapping from source data

## Architecture

```
numbers_dont_lie/
├── models.py              # ColumnStats, TableStats dataclasses
├── generator.py           # Pandas/NumPy synthetic data generator
├── orchestrator.py        # Pipeline coordination
├── readers/
│   ├── __init__.py        # BaseStatsReader, BaseWriter ABCs
│   ├── postgres.py        # PostgreSQL reader & writer
│   ├── mysql.py           # MySQL reader & writer
│   └── sqlite.py          # SQLite reader & writer
└── gui/
    ├── app.py             # Entry point
    ├── main_window.py     # Main GUI window
    └── connection_dialog.py  # Connection configuration
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

MIT — see [LICENSE](LICENSE).

Run all cells to execute the pipeline

The orchestrator will discover all tables in the source path, extract their statistics, generate synthetic data, and write to the destination.

## Configuration Options

### Getting Your Lakehouse ABFSS Path

1. Open your lakehouse in Microsoft Fabric
2. Click on the "..." menu and select "Properties"
3. Copy the ABFSS path (format: `abfss://workspace-guid@onelake.dfs.fabric.microsoft.com/lakehouse-guid/`)
4. Add `/Tables/` to the end to process all tables

### Source Path Formats

Entire lakehouse (ABFSS): `abfss://workspace-guid@onelake.dfs.fabric.microsoft.com/lakehouse-guid/Tables/`
Specific schema/folder (ABFSS): `abfss://workspace-guid@onelake.dfs.fabric.microsoft.com/lakehouse-guid/Tables/bronze_afas/`
Using catalog names: `production_lakehouse.schema` or `production_lakehouse.schema.table`

### Scale Factor

1.0 = Same number of rows as source
0.1 = 10% for quick testing
2.0 = Double the data for load testing

### V-Order Optimization

Set `ENABLE_VORDER = True` to optimize read performance in Fabric (recommended)

## How It Works

### Privacy Guarantee

The pipeline guarantees complete privacy through:

No actual data rows are ever loaded - All input comes from Delta _delta_log metadata or DESCRIBE commands
Distributed operations only - No collect() calls on source data
Approximate aggregations - APPROX_COUNT_DISTINCT avoids full table scans
Random generation - No deterministic mapping from source data

This ensures that even with access to both source and synthetic data, no individual source record can be reconstructed.

### Generation Strategy

For each column type:

Numeric: Uniform random distribution between min/max
Categorical: Mock labels (Category_1, Category_2, etc.) matching distinct count
Dates/Timestamps: Random values within observed range
UUIDs: Fresh random UUIDs
Nulls: Injected to match source null ratio

All operations use pyspark.sql.functions for distributed processing.

## Validation

The notebook includes validation cells to compare source vs synthetic statistics:

Row counts and scale factor verification
Distinct count comparison per column
Null count and ratio comparison
Min/max value range verification

## Extensibility

The clean separation between statistics interface and reader implementation makes it easy to add support for other data sources.

### Adding SQL Server Support

1. Create SqlServerStatsReader class
2. Query sys.dm_db_stats_properties and DBCC SHOW_STATISTICS
3. Return TableStats object
4. Reuse the same generator and writer

See section 10 in the notebook for detailed examples.

## Contributing

Contributions welcome! Areas of interest:

Additional data source readers (SQL Server, PostgreSQL, Snowflake)
More sophisticated distribution modeling (normal, exponential)
Data quality rules and constraints
Performance optimizations
Documentation improvements

## License

MIT License - see LICENSE file for details

## Authors

Created for the Microsoft Fabric community

## Acknowledgments

Built on Microsoft Fabric and PySpark
Inspired by the need for privacy-preserving test data in enterprise environments
