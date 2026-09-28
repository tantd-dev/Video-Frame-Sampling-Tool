"""
Video Frame Sampling Tool — Entry Point

A desktop GUI tool for extracting frames from video files using
multiple sampling strategies: Simple, Scene-change, Smart (heuristic),
and Hierarchical.

Usage:
    python main.py
"""

import sys
import os

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from config import APP_NAME
from ui.main_window import MainWindow


def main():
    # Enable high-DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
