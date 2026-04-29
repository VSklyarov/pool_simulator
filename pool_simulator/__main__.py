"""Entry point for `python -m pool_simulator`."""

import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from pool_simulator.core.paths import get_bundle_dir
from pool_simulator.ui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Pool Simulator")

    icon_path = get_bundle_dir() / "assets" / "icon.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
