"""Main window for the Numbers Don't Lie GUI."""

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from numbers_dont_lie.gui.connection_dialog import ConnectionDialog
from numbers_dont_lie.orchestrator import Orchestrator, TableResult
from numbers_dont_lie.readers import BaseStatsReader, BaseWriter


def _create_reader(config: dict) -> BaseStatsReader:
    """Factory to create reader from config dict."""
    db_type = config["type"]
    if db_type == "postgresql":
        from numbers_dont_lie.readers.postgres import PostgresStatsReader
        return PostgresStatsReader(
            host=config["host"], port=config["port"],
            database=config["database"], user=config["user"], password=config["password"],
        )
    elif db_type == "mysql":
        from numbers_dont_lie.readers.mysql import MySQLStatsReader
        return MySQLStatsReader(
            host=config["host"], port=config["port"],
            database=config["database"], user=config["user"], password=config["password"],
        )
    elif db_type == "sqlite":
        from numbers_dont_lie.readers.sqlite import SQLiteStatsReader
        return SQLiteStatsReader(database_path=config["database_path"])
    elif db_type == "odbc":
        from numbers_dont_lie.readers.odbc import ODBCStatsReader
        return ODBCStatsReader(connection_string=config["connection_string"])
    else:
        raise ValueError(f"Unknown database type: {db_type}")


def _create_writer(config: dict) -> BaseWriter:
    """Factory to create writer from config dict."""
    db_type = config["type"]
    if db_type == "postgresql":
        from numbers_dont_lie.readers.postgres import PostgresWriter
        return PostgresWriter(
            host=config["host"], port=config["port"],
            database=config["database"], user=config["user"], password=config["password"],
        )
    elif db_type == "mysql":
        from numbers_dont_lie.readers.mysql import MySQLWriter
        return MySQLWriter(
            host=config["host"], port=config["port"],
            database=config["database"], user=config["user"], password=config["password"],
        )
    elif db_type == "sqlite":
        from numbers_dont_lie.readers.sqlite import SQLiteWriter
        return SQLiteWriter(database_path=config["database_path"])
    elif db_type == "odbc":
        from numbers_dont_lie.readers.odbc import ODBCWriter
        return ODBCWriter(connection_string=config["connection_string"])
    else:
        raise ValueError(f"Unknown database type: {db_type}")


class GenerationWorker(QThread):
    """Background thread for running the generation pipeline."""

    progress = Signal(int, int, str)
    finished = Signal(list)
    error = Signal(str)
    log = Signal(str)

    def __init__(self, source_config, dest_config, scale_factor, seed=None):
        super().__init__()
        self.source_config = source_config
        self.dest_config = dest_config
        self.scale_factor = scale_factor
        self.seed = seed

    def run(self):
        try:
            reader = _create_reader(self.source_config)
            writer = _create_writer(self.dest_config)

            reader.connect()
            writer.connect()

            self.log.emit(f"Connected to source ({self.source_config['type']})")
            self.log.emit(f"Connected to destination ({self.dest_config['type']})")

            orchestrator = Orchestrator(
                reader=reader,
                writer=writer,
                scale_factor=self.scale_factor,
                seed=self.seed,
            )

            def on_progress(current, total, table_name):
                self.progress.emit(current, total, table_name)
                self.log.emit(f"[{current + 1}/{total}] Processing {table_name}...")

            dest_schema = self.dest_config.get("schema") or "synthetic"
            results = orchestrator.run(
                schema=self.source_config.get("schema"),
                dest_schema=dest_schema,
                progress_callback=on_progress,
            )

            reader.disconnect()
            writer.disconnect()

            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Numbers Don't Lie - Synthetic Data Generator")
        self.setMinimumSize(900, 600)

        self.source_config = None
        self.dest_config = None
        self.worker = None

        self._setup_ui()
        self._update_status()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Top: Connection configuration
        conn_group = QGroupBox("Connections")
        conn_layout = QHBoxLayout(conn_group)

        # Source
        source_layout = QVBoxLayout()
        self.source_label = QLabel("Source: Not configured")
        source_layout.addWidget(self.source_label)
        self.source_btn = QPushButton("Configure Source...")
        self.source_btn.clicked.connect(self._configure_source)
        source_layout.addWidget(self.source_btn)
        conn_layout.addLayout(source_layout)

        # Destination
        dest_layout = QVBoxLayout()
        self.dest_label = QLabel("Destination: Not configured")
        dest_layout.addWidget(self.dest_label)
        self.dest_btn = QPushButton("Configure Destination...")
        self.dest_btn.clicked.connect(self._configure_destination)
        dest_layout.addWidget(self.dest_btn)
        conn_layout.addLayout(dest_layout)

        main_layout.addWidget(conn_group)

        # Middle: Options
        options_group = QGroupBox("Options")
        options_layout = QHBoxLayout(options_group)

        options_layout.addWidget(QLabel("Scale Factor:"))
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.01, 100.0)
        self.scale_spin.setValue(1.0)
        self.scale_spin.setSingleStep(0.1)
        options_layout.addWidget(self.scale_spin)

        options_layout.addStretch()

        self.run_btn = QPushButton("Generate Synthetic Data")
        self.run_btn.setEnabled(False)
        self.run_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 8px 16px; }")
        self.run_btn.clicked.connect(self._run_generation)
        options_layout.addWidget(self.run_btn)

        main_layout.addWidget(options_group)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # Bottom: Splitter with results table and log
        splitter = QSplitter(Qt.Vertical)

        # Results table
        self.results_table = QTableWidget(0, 4)
        self.results_table.setHorizontalHeaderLabels(["Table", "Status", "Source Rows", "Generated Rows"])
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        splitter.addWidget(self.results_table)

        # Log output
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        splitter.addWidget(self.log_text)

        main_layout.addWidget(splitter)

        # Status bar
        self.setStatusBar(QStatusBar())

    def _configure_source(self):
        dialog = ConnectionDialog(self, mode="source")
        if dialog.exec() == ConnectionDialog.Accepted:
            self.source_config = dialog.get_connection_config()
            db_type = self.source_config["type"]
            if db_type == "sqlite":
                self.source_label.setText(f"Source: SQLite - {self.source_config['database_path']}")
            else:
                self.source_label.setText(
                    f"Source: {db_type.title()} - {self.source_config.get('host')}:{self.source_config.get('port')}/{self.source_config.get('database')}"
                )
            self._update_status()

    def _configure_destination(self):
        dialog = ConnectionDialog(self, mode="destination")
        if dialog.exec() == ConnectionDialog.Accepted:
            self.dest_config = dialog.get_connection_config()
            db_type = self.dest_config["type"]
            if db_type == "sqlite":
                self.dest_label.setText(f"Destination: SQLite - {self.dest_config['database_path']}")
            else:
                self.dest_label.setText(
                    f"Destination: {db_type.title()} - {self.dest_config.get('host')}:{self.dest_config.get('port')}/{self.dest_config.get('database')}"
                )
            self._update_status()

    def _update_status(self):
        ready = self.source_config is not None and self.dest_config is not None
        self.run_btn.setEnabled(ready)
        if ready:
            self.statusBar().showMessage("Ready to generate synthetic data")
        else:
            missing = []
            if not self.source_config:
                missing.append("source")
            if not self.dest_config:
                missing.append("destination")
            self.statusBar().showMessage(f"Configure {' and '.join(missing)} to begin")

    def _run_generation(self):
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.results_table.setRowCount(0)
        self.log_text.clear()

        self.log_text.append("Starting synthetic data generation...")

        self.worker = GenerationWorker(
            source_config=self.source_config,
            dest_config=self.dest_config,
            scale_factor=self.scale_spin.value(),
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.log.connect(self._on_log)
        self.worker.start()

    def _on_progress(self, current: int, total: int, table_name: str):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.statusBar().showMessage(f"Processing {table_name} ({current + 1}/{total})")

    def _on_log(self, message: str):
        self.log_text.append(message)

    def _on_finished(self, results: list):
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)

        # Populate results table
        self.results_table.setRowCount(len(results))
        success = 0
        for i, result in enumerate(results):
            self.results_table.setItem(i, 0, QTableWidgetItem(result.table))
            status_item = QTableWidgetItem(result.status)
            if result.status == "SUCCESS":
                success += 1
            self.results_table.setItem(i, 1, status_item)
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{result.source_rows:,}"))
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{result.generated_rows:,}"))

        self.log_text.append(f"\nComplete! {success}/{len(results)} tables generated successfully.")
        self.statusBar().showMessage(f"Done - {success}/{len(results)} tables succeeded")

    def _on_error(self, error_message: str):
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        self.log_text.append(f"\nERROR: {error_message}")
        QMessageBox.critical(self, "Generation Failed", error_message)
