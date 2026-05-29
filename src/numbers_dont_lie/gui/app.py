"""Entry point for the Numbers Don't Lie GUI application."""

import sys


def main():
    from PySide6.QtWidgets import QApplication

    from numbers_dont_lie.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Numbers Don't Lie")
    app.setOrganizationName("NumbersDontLie")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
