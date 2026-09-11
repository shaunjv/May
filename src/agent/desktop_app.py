"""Launch entry point for the local PySide6 desktop agent."""
import argparse
import asyncio
import os
import sys
from pathlib import Path

from agent.config import Settings
from agent.desktop.presenter import DesktopPresenter
from agent.desktop.runtime import DesktopRuntime
from agent.desktop.ui import launch


def main() -> None:
    # A double-clicked packaged executable starts in an unpredictable working
    # directory. Keep the local .env beside the executable and load it first.
    launch_dir = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path.cwd()
    os.chdir(launch_dir)
    parser = argparse.ArgumentParser(description="Launch the Personal AI Agent desktop app.")
    parser.add_argument("workspace", nargs="?", default=None)
    args = parser.parse_args()
    if args.workspace is None:
        from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
        app = QApplication.instance() or QApplication([])
        args.workspace = QFileDialog.getExistingDirectory(None, "Choose the workspace for this conversation")
        if not args.workspace:
            return
    try:
        runtime = asyncio.run(DesktopRuntime.create(Path(args.workspace), Settings()))
    except Exception as error:
        from PySide6.QtWidgets import QApplication, QMessageBox
        app = QApplication.instance() or QApplication([])
        QMessageBox.critical(None, "Personal AI Agent could not start", str(error))
        return
    raise SystemExit(launch(DesktopPresenter(runtime)))


if __name__ == "__main__":
    main()
