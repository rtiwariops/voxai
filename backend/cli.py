import subprocess
import sys
from pathlib import Path

MIN_NODE_MAJOR = 14  # minimum supported Node.js major version

def check_node_version():
    try:
        out = subprocess.check_output(["node", "--version"], text=True).strip()
        # out is like "v12.13.0"
        ver = out.lstrip("v").split(".")
        major = int(ver[0])
    except Exception:
        print("❌ Could not determine your Node.js version. Please install Node.js ≥14.", file=sys.stderr)
        sys.exit(1)

    if major < MIN_NODE_MAJOR:
        print(f"❌ Your Node.js version is {out}. AuroraAI requires Node.js ≥{MIN_NODE_MAJOR}.", file=sys.stderr)
        print("   If you use nvm, run:", file=sys.stderr)
        print("     nvm install 16      # or `nvm install stable`", file=sys.stderr)
        print("     nvm use 16", file=sys.stderr)
        sys.exit(1)

def main():
    base   = Path(__file__).parent.parent.resolve()
    ui_dir = base / "electron"

    # 0) Check Node version
    check_node_version()

    # 1) Start backend
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "backend.core"],
        cwd=str(base),
    )
    print("✅ VoxAI backend started.")

    # 2) Launch Electron
    try:
        subprocess.run(
            ["npx", "electron", str(ui_dir)],
            check=True,
            cwd=str(ui_dir),
        )
    except FileNotFoundError:
        print("❌ Could not find 'npx'. Did you run 'npm install' in electron/?", file=sys.stderr)
        backend_proc.terminate()
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ Electron exited with code {e.returncode}", file=sys.stderr)
    finally:
        backend_proc.terminate()
        backend_proc.wait()
        print("✅ VoxAI backend stopped.")

if __name__ == "__main__":
    main()
