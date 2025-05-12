import subprocess
import sys
import os
from pathlib import Path

MIN_NODE_MAJOR = 14  # require Node.js ≥14

def check_node_version():
    try:
        out = subprocess.check_output(["node", "--version"], text=True).strip()
        major = int(out.lstrip("v").split(".")[0])
        if major < MIN_NODE_MAJOR:
            raise ValueError
    except Exception:
        print("❌ Node.js ≥14 is required. Please install or upgrade Node.js.", file=sys.stderr)
        sys.exit(1)

def ensure_electron_installed(ui_dir: Path):
    """Run 'npm install' in electron/ if needed so that 'npx electron' will work."""
    bin_path = ui_dir / "node_modules" / ".bin" / "electron"
    if not bin_path.exists():
        print("ℹ️ Installing Electron dependencies…", file=sys.stderr)
        try:
            subprocess.run(["npm", "install"], check=True, cwd=str(ui_dir))
        except Exception as e:
            print(f"❌ Failed to run 'npm install' in {ui_dir}: {e}", file=sys.stderr)
            sys.exit(1)

def launch_electron(ui_dir: Path):
    """Try local npx electron, then global electron."""
    # First ensure local deps
    ensure_electron_installed(ui_dir)

    # Try npx
    for cmd in (["npx", "electron", str(ui_dir)], ["electron", str(ui_dir)]):
        try:
            subprocess.run(cmd, check=True, cwd=str(ui_dir))
            return
        except FileNotFoundError:
            continue
        except subprocess.CalledProcessError as e:
            print(f"❌ Electron exited with code {e.returncode}", file=sys.stderr)
            return

    print("❌ Could not launch Electron (tried 'npx electron' & 'electron').", file=sys.stderr)
    sys.exit(1)

def main():
    # 1) Ensure Node.js version is sufficient
    check_node_version()

    base   = Path(__file__).parent.parent.resolve()
    ui_dir = base / "electron"

    # 2) Start the Python backend
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "backend.core"],
        cwd=str(base),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    print("✅ VoxAI backend started.")

    # forward backend logs to our stdout
    def forward_logs():
        for line in backend_proc.stdout:
            sys.stdout.write(line)
    import threading
    threading.Thread(target=forward_logs, daemon=True).start()

    # 3) Launch Electron UI
    launch_electron(ui_dir)

    # 4) Cleanup
    backend_proc.terminate()
    backend_proc.wait()
    print("✅ VoxAI backend stopped.")

if __name__ == "__main__":
    main()
