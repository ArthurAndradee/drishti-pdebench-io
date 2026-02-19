import subprocess
import re
import statistics
from pathlib import Path
from collections import defaultdict
import sys
import shutil

# -------------------------
# Configuration
# -------------------------
RESULTS_ROOT = Path("/home/users/aadsilva/ic/erad-2026/darshan-logs/results")

# -------------------------
# Helpers
# -------------------------

def get_darshan_output(file_path):
    """Safely runs darshan-parser and returns stdout."""
    try:
        # We use --show-incomplete to handle logs from killed/timed-out jobs
        result = subprocess.run(
            ["darshan-parser", "--show-incomplete", str(file_path)],
            capture_output=True, text=True, check=False
        )
        return result.stdout
    except Exception as e:
        return ""

def parse_darshan_log(file_path: Path):
    metrics = {
        "run_time": 0.0,
        "bytes_read": 0,
        "bytes_written": 0
    }

    output = get_darshan_output(file_path)
    if not output:
        return metrics

    # 1. Extract Runtime (from header)
    rt_match = re.search(r"^#\s*run time:\s*([\d\.]+)", output, re.MULTILINE)
    if rt_match:
        metrics["run_time"] = float(rt_match.group(1))

    # 2. Extract I/O Counters (Iterate all lines)
    # Logic: Look for any token containing "BYTES_READ" or "BYTES_WRITTEN"
    # Format typically: <Module> <Rank> <ID> <CounterName> <Value> ...
    for line in output.splitlines():
        # Optimization: Skip lines that don't mention bytes
        if "BYTES_" not in line:
            continue
            
        parts = line.split()
        
        # Iterate through tokens to find the counter name
        for i, part in enumerate(parts):
            # We check length > i+1 to ensure a value exists after the counter name
            if i + 1 >= len(parts):
                break

            # Check for READ counters (POSIX_BYTES_READ, MPIIO_BYTES_READ, etc.)
            if "BYTES_READ" in part:
                try:
                    metrics["bytes_read"] += int(parts[i+1])
                except ValueError: pass
            
            # Check for WRITE counters
            elif "BYTES_WRITTEN" in part:
                try:
                    metrics["bytes_written"] += int(parts[i+1])
                except ValueError: pass

    return metrics

def format_bytes(size):
    power_labels = {0 : 'B', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB', 5: 'PB'}
    n = 0
    while size > 1024:
        size /= 1024
        n += 1
    return f"{size:.2f} {power_labels.get(n, '')}"

# -------------------------
# Main Logic
# -------------------------
def main():
    if not RESULTS_ROOT.exists():
        print(f"❌ Error: Root directory {RESULTS_ROOT} does not exist.")
        return

    # Structure: data[model][dataset][optimization] = list_of_metric_dicts
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    print(f"📂 Scanning directory: {RESULTS_ROOT}")
    
    found_any = False

    for model_dir in sorted(RESULTS_ROOT.iterdir()):
        if not model_dir.is_dir(): continue
        for dataset_dir in sorted(model_dir.iterdir()):
            if not dataset_dir.is_dir(): continue
            for opt_dir in sorted(dataset_dir.iterdir()):
                if not opt_dir.is_dir(): continue
                
                logs = sorted(opt_dir.glob("*.darshan"))
                if not logs: continue
                
                found_any = True
                print(f"  Processing {model_dir.name}/{dataset_dir.name}/{opt_dir.name} ({len(logs)} logs)...")

                for log_file in logs:
                    m = parse_darshan_log(log_file)
                    
                    # Filter out completely empty/failed parses
                    if m["run_time"] > 0 or m["bytes_read"] > 0 or m["bytes_written"] > 0:
                        data[model_dir.name][dataset_dir.name][opt_dir.name].append(m)

    if not found_any:
        print("❌ No .darshan logs found.")
        return

    # -------------------------
    # Reporting
    # -------------------------
    print("\n" + "="*95)
    print(f"{'CONFIGURATION':<50} | {'RUNTIME (s)':<18} | {'IO VOLUME'}")
    print(f"{'Model / Dataset / Opt':<50} | {'Avg ± Std':<18} | {'Avg R/W'}")
    print("="*95)

    for model, datasets in data.items():
        for dataset, opts in datasets.items():
            for opt, metric_list in opts.items():
                if not metric_list: continue

                runtimes = [m["run_time"] for m in metric_list]
                reads = [m["bytes_read"] for m in metric_list]
                writes = [m["bytes_written"] for m in metric_list]

                avg_time = statistics.mean(runtimes)
                std_time = statistics.stdev(runtimes) if len(runtimes) > 1 else 0.0
                
                avg_read = statistics.mean(reads)
                avg_write = statistics.mean(writes)

                label = f"{model}/{dataset}/{opt}"
                # Truncate label if too long for cleaner table
                if len(label) > 48: label = label[:45] + "..."

                print(f"{label:<50} | {avg_time:.2f} ± {std_time:.2f}   | R: {format_bytes(avg_read)}")
                print(f"{'':<50} | {'(n=' + str(len(runtimes)) + ')':<18} | W: {format_bytes(avg_write)}")
                print("-" * 95)

if __name__ == "__main__":
    main()