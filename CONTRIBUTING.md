# Contributing to Numbers Don't Lie

Thank you for your interest in contributing to Numbers Don't Lie.

## How to Contribute

### Reporting Issues

Use the GitHub issue tracker to report bugs or suggest features
Include detailed steps to reproduce bugs
Provide sample configurations and error messages when possible

### Pull Requests

Fork the repository and create a feature branch
Follow the existing code style and structure
Add tests for new functionality
Update documentation as needed
Submit a pull request with a clear description of changes

## Development Areas

### High Priority

SQL Server statistics reader implementation
PostgreSQL statistics reader
Snowflake statistics reader
Advanced distribution modeling (normal, log-normal, exponential)
Data quality constraints (foreign keys, check constraints)

### Documentation

Usage examples for different scenarios
Performance tuning guide
Architecture diagrams
Video tutorials

### Testing

Unit tests for generators
Integration tests with real Delta tables
Performance benchmarks
Privacy validation tests

## Code Style

Follow PEP 8 for Python code
Use type hints where appropriate
Add docstrings to all public classes and methods
Keep functions focused and single-purpose
Use meaningful variable names

## Architecture Principles

Maintain strict separation between statistics interface and implementations
Never load actual data rows into memory
Use distributed operations (PySpark) for all data processing
Preserve privacy guarantees in all new features
Keep extensibility as a core design goal

## Testing

Test with various Delta table schemas
Verify statistical accuracy of generated data
Validate privacy guarantees
Test error handling and edge cases
Benchmark performance with large datasets

## Questions

Open a GitHub discussion for questions about architecture or implementation
Use issues for specific bugs or feature requests

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
