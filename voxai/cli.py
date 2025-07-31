import os
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

from .config import MIN_NODE_MAJOR_VERSION, EnvVars

MIN_NODE_MAJOR = MIN_NODE_MAJOR_VERSION

def check_node_version() -> None:
    """Check if Node.js version meets minimum requirements.
    
    Raises:
        SystemExit: If Node.js is not installed or version is too old.
    """
    try:
        out = subprocess.check_output(["node", "--version"], text=True).strip()
        if int(out.lstrip("v").split(".")[0]) < MIN_NODE_MAJOR:
            raise ValueError
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, OSError) as e:
        print(f"❌ Node.js ≥{MIN_NODE_MAJOR} is required. Error: {e}", file=sys.stderr)
        sys.exit(1)

def ensure_ui_bootstrapped(ui_dir: Path) -> None:
    """Ensure Electron UI dependencies are installed.
    
    Args:
        ui_dir: Path to the Electron UI directory.
        
    Raises:
        SystemExit: If npm install fails.
    """
    # Only bootstrap once: if node_modules doesn't exist
    if not (ui_dir / "node_modules").exists():
        print("ℹ️ Bootstrapping Electron UI (npm install)...", file=sys.stderr)
        try:
            subprocess.run(["npm", "install"], check=True, cwd=str(ui_dir))
        except subprocess.CalledProcessError as e:
            print(f"❌ npm install failed ({e.returncode}).", file=sys.stderr)
            sys.exit(1)

def launch_ui(ui_dir: Path) -> NoReturn:
    """Launch the Electron UI application.
    
    Args:
        ui_dir: Path to the Electron UI directory.
        
    Raises:
        SystemExit: If Electron fails to launch or exits with an error.
    """
    ensure_ui_bootstrapped(ui_dir)
    
    # Pass the current Python executable to Electron
    env = os.environ.copy()
    env[EnvVars.PYTHON_EXE] = sys.executable
    
    try:
        subprocess.run(["npx", "electron", str(ui_dir)], check=True, cwd=str(ui_dir), env=env)
    except FileNotFoundError:
        # fallback to global electron
        try:
            subprocess.run(["electron", str(ui_dir)], check=True, cwd=str(ui_dir), env=env)
        except (FileNotFoundError, subprocess.CalledProcessError, OSError) as e:
            print(f"❌ Could not launch Electron (npx or global): {e}", file=sys.stderr)
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ Electron exited with code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)

def main() -> None:
    """Main entry point for VoxAI application.
    
    Coordinates the startup sequence:
    1. Validates Node.js version requirements
    2. Starts the Python backend process
    3. Sets up log forwarding from backend to frontend
    4. Launches the Electron UI (blocking call)
    5. Cleans up backend process on UI exit
    """
    check_node_version()

    # Locate the installed package root
    pkg_root = Path(__file__).parent
    ui_dir = pkg_root / "electron"

    # 1) Start the Python backend
    backend = subprocess.Popen(
        [sys.executable, "-m", "voxai.core"],
        cwd=str(pkg_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    print("✅ VoxAI backend started.")

    # 2) Forward backend logs
    import threading
    def _fwd() -> None:
        """Forward backend stdout to main process stdout."""
        for line in backend.stdout:
            sys.stdout.write(line)
    threading.Thread(target=_fwd, daemon=True).start()

    # 3) Launch the Electron UI
    launch_ui(ui_dir)

    # 4) Cleanup
    backend.terminate()
    backend.wait()
    print("✅ VoxAI backend stopped.")

if __name__ == "__main__":
    main()