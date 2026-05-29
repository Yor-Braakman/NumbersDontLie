"""Connection dialog for configuring database connections."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


class ConnectionDialog(QDialog):
    """Dialog for configuring a database connection."""

    def __init__(self, parent=None, mode: str = "source"):
        super().__init__(parent)
        self.setWindowTitle(f"Configure {'Source' if mode == 'source' else 'Destination'} Connection")
        self.setMinimumWidth(450)
        self.mode = mode
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Database type selector
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Database Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["PostgreSQL", "MySQL", "SQLite", "ODBC (SQL Server, Snowflake, etc.)"])
        self.type_combo.currentIndexChanged.connect(self._on_type_changed)
        type_layout.addWidget(self.type_combo)
        layout.addLayout(type_layout)

        # Stacked widget for different connection forms
        self.stack = QStackedWidget()

        # PostgreSQL form
        self.pg_widget = self._create_server_form("5432")
        self.stack.addWidget(self.pg_widget)

        # MySQL form
        self.mysql_widget = self._create_server_form("3306")
        self.stack.addWidget(self.mysql_widget)

        # SQLite form
        self.sqlite_widget = self._create_sqlite_form()
        self.stack.addWidget(self.sqlite_widget)

        # ODBC form
        self.odbc_widget = self._create_odbc_form()
        self.stack.addWidget(self.odbc_widget)

        layout.addWidget(self.stack)

        # Schema filter (optional)
        schema_layout = QHBoxLayout()
        schema_layout.addWidget(QLabel("Schema (optional):"))
        self.schema_edit = QLineEdit()
        self.schema_edit.setPlaceholderText("Leave empty for all schemas")
        schema_layout.addWidget(self.schema_edit)
        layout.addLayout(schema_layout)

        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _create_server_form(self, default_port: str) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)

        host_edit = QLineEdit("localhost")
        form.addRow("Host:", host_edit)

        port_spin = QSpinBox()
        port_spin.setRange(1, 65535)
        port_spin.setValue(int(default_port))
        form.addRow("Port:", port_spin)

        db_edit = QLineEdit()
        db_edit.setPlaceholderText("Database name")
        form.addRow("Database:", db_edit)

        user_edit = QLineEdit()
        form.addRow("User:", user_edit)

        pass_edit = QLineEdit()
        pass_edit.setEchoMode(QLineEdit.Password)
        form.addRow("Password:", pass_edit)

        return widget

    def _create_sqlite_form(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        file_layout = QHBoxLayout()
        self.sqlite_path_edit = QLineEdit()
        self.sqlite_path_edit.setPlaceholderText("Path to .db file")
        file_layout.addWidget(self.sqlite_path_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_sqlite)
        file_layout.addWidget(browse_btn)

        layout.addLayout(file_layout)
        layout.addStretch()

        return widget

    def _browse_sqlite(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SQLite Database", "", "SQLite Files (*.db *.sqlite *.sqlite3);;All Files (*)"
        )
        if path:
            self.sqlite_path_edit.setText(path)

    def _create_odbc_form(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        layout.addWidget(QLabel("ODBC Connection String:"))
        self.odbc_connstr_edit = QLineEdit()
        self.odbc_connstr_edit.setPlaceholderText(
            "DRIVER={ODBC Driver 18 for SQL Server};SERVER=localhost;DATABASE=mydb;UID=sa;PWD=pass"
        )
        layout.addWidget(self.odbc_connstr_edit)

        layout.addWidget(QLabel(
            "Examples:\n"
            "  SQL Server: DRIVER={ODBC Driver 18 for SQL Server};SERVER=srv;DATABASE=db;UID=u;PWD=p\n"
            "  Snowflake: DRIVER={Snowflake};SERVER=acct.snowflakecomputing.com;DATABASE=db;UID=u;PWD=p\n"
            "  Trusted: DRIVER={ODBC Driver 18 for SQL Server};SERVER=srv;DATABASE=db;Trusted_Connection=yes"
        ))
        layout.addStretch()

        return widget

    def _on_type_changed(self, index: int):
        self.stack.setCurrentIndex(index)

    def get_connection_config(self) -> dict:
        """Return the connection configuration as a dictionary."""
        raw_type = self.type_combo.currentText()
        if raw_type.startswith("ODBC"):
            db_type = "odbc"
        else:
            db_type = raw_type.lower()
        config = {"type": db_type, "schema": self.schema_edit.text().strip() or None}

        if db_type == "sqlite":
            config["database_path"] = self.sqlite_path_edit.text().strip()
        elif db_type == "odbc":
            config["connection_string"] = self.odbc_connstr_edit.text().strip()
        else:
            widget = self.stack.currentWidget()
            form = widget.layout()

            config["host"] = form.itemAt(0, QFormLayout.FieldRole).widget().text()
            config["port"] = form.itemAt(1, QFormLayout.FieldRole).widget().value()
            config["database"] = form.itemAt(2, QFormLayout.FieldRole).widget().text()
            config["user"] = form.itemAt(3, QFormLayout.FieldRole).widget().text()
            config["password"] = form.itemAt(4, QFormLayout.FieldRole).widget().text()

        return config
