"""Launch the Streamlit dashboard with the active Python interpreter."""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    app_path = project_root / "dashboard" / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path)],
        cwd=project_root,
        check=True,
    )


if __name__ == "__main__":
    main()
