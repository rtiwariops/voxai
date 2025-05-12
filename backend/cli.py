import subprocess
from pathlib import Path

def main():
    # Launch the Electron UI bundled in the package
    base = Path(__file__).parent.parent.resolve()
    ui_dir = base / "electron"
    subprocess.run(["npx", "electron", str(ui_dir)], check=True)

if __name__ == "__main__":
    main()
