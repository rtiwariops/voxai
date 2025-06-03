import subprocess
import sys
from pathlib import Path

MIN_NODE_MAJOR = 14

def check_node_version():
    try:
        out = subprocess.check_output(["node", "--version"], text=True).strip()
        if int(out.lstrip("v").split(".")[0]) < MIN_NODE_MAJOR:
            raise ValueError
    except Exception:
        print(f"❌ Node.js ≥{MIN_NODE_MAJOR} is required.", file=sys.stderr)
        sys.exit(1)

def ensure_ui_bootstrapped(ui_dir: Path):
    # Only bootstrap once: if node_modules doesn't exist
    if not (ui_dir / "node_modules").exists():
        print("ℹ️ Bootstrapping Electron UI (npm install)...", file=sys.stderr)
        try:
            subprocess.run(["npm", "install"], check=True, cwd=str(ui_dir))
        except subprocess.CalledProcessError as e:
            print(f"❌ npm install failed ({e.returncode}).", file=sys.stderr)
            sys.exit(1)

def launch_ui(ui_dir: Path):
    ensure_ui_bootstrapped(ui_dir)
    try:
        subprocess.run(["npx", "electron", str(ui_dir)], check=True, cwd=str(ui_dir))
    except FileNotFoundError:
        # fallback to global electron
        try:
            subprocess.run(["electron", str(ui_dir)], check=True, cwd=str(ui_dir))
        except Exception:
            print("❌ Could not launch Electron (npx or global).", file=sys.stderr)
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ Electron exited with code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)

def main():
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
    def _fwd():
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