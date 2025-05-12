import subprocess
import sys
from pathlib import Path

MIN_NODE_MAJOR = 14

def check_node_version():
    try:
        out = subprocess.check_output(["node","--version"], text=True).strip()
        major = int(out.lstrip("v").split(".")[0])
        if major < MIN_NODE_MAJOR:
            raise ValueError
    except Exception:
        print(f"❌ Node.js ≥{MIN_NODE_MAJOR} is required.", file=sys.stderr)
        sys.exit(1)

def ensure_electron(ui_dir: Path):
    bin_e = ui_dir / "node_modules" / ".bin" / "electron"
    if not bin_e.exists():
        print("ℹ️ Installing Electron…", file=sys.stderr)
        subprocess.run(["npm","install"], check=True, cwd=str(ui_dir))

def launch_electron(ui_dir: Path):
    ensure_electron(ui_dir)
    for cmd in (["npx","electron",str(ui_dir)], ["electron",str(ui_dir)]):
        try:
            subprocess.run(cmd, check=True, cwd=str(ui_dir))
            return
        except FileNotFoundError:
            continue
        except subprocess.CalledProcessError as e:
            print(f"❌ Electron exited {e.returncode}", file=sys.stderr)
            return
    print("❌ Could not launch Electron.", file=sys.stderr)
    sys.exit(1)

def main():
    check_node_version()
    base   = Path(__file__).parent.resolve()
    ui_dir = base / "electron"

    # start backend
    backend = subprocess.Popen(
        [sys.executable, "-m", "voxai.core"],
        cwd=str(base.parent),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    print("✅ VoxAI backend started.")

    # forward logs
    import threading
    def _fwd():
        for line in backend.stdout:
            sys.stdout.write(line)
    threading.Thread(target=_fwd, daemon=True).start()

    # launch UI
    launch_electron(ui_dir)

    # cleanup
    backend.terminate()
    backend.wait()
    print("✅ VoxAI backend stopped.")

if __name__ == "__main__":
    main()
