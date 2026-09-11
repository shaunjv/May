"""Build a double-clickable Windows executable without bundling secrets."""

from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"


def main() -> None:
    command = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onefile", "--windowed",
        "--name", "PersonalAIAgent", "--paths", str(ROOT / "src"),
        "--collect-all", "PySide6", "--collect-all", "openai", str(ROOT / "src" / "agent" / "desktop_app.py"),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    shutil.copy2(ROOT / ".env.example", DIST / ".env.example")
    print(f"Created {DIST / 'PersonalAIAgent.exe'}")
    print("Create a .env beside the executable from .env.example; never copy an API key into source control.")


if __name__ == "__main__":
    main()
