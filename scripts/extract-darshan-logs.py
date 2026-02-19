import subprocess
from pathlib import Path
import shutil

# -------------------------
# Configuration
# -------------------------
DARSHAN_FOLDER = Path(
    "/home/users/aadsilva/ic/erad-2026/darshan-logs/2026/2/17"
)

OUTPUT_BASE = DARSHAN_FOLDER / "filtered_logs"

COMMANDS = [
    "train_models_forward_STD",
    "train_models_forward_ARB",
    "train_models_forward_BRO",
    "train_models_forward_MPI",
]


# -------------------------
# Helpers
# -------------------------
def get_command(darshan_file: Path) -> str:
    """Extract executable line from darshan log."""
    result = subprocess.run(
        ["darshan-parser", "--show-incomplete", str(darshan_file)],
        capture_output=True,
        text=True,
    )

    combined = (result.stdout or "") + "\n" + (result.stderr or "")

    for line in combined.splitlines():
        if line.lower().startswith("# exe:"):
            return line.split(":", 1)[1].strip()

    return ""


# -------------------------
# Main logic
# -------------------------
def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # Create destination folders
    dest_dirs = {}
    for cmd in COMMANDS:
        d = OUTPUT_BASE / cmd
        d.mkdir(exist_ok=True)
        dest_dirs[cmd] = d

    darshan_files = sorted(DARSHAN_FOLDER.glob("*.darshan"))
    print(f"🔍 Found {len(darshan_files)} darshan files")

    matched = 0

    for darshan_file in darshan_files:
        cmdline = get_command(darshan_file)

        for cmd in COMMANDS:
            if cmd in cmdline:
                dest = dest_dirs[cmd] / darshan_file.name
                shutil.copy2(darshan_file, dest)
                matched += 1

                print(f"✅ {darshan_file.name} → {cmd}")
                break

    print(f"\n📦 Total matched logs: {matched}")
    print(f"📁 Output directory: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
