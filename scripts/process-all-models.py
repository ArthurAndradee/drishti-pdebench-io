import os
import glob
import shutil
import subprocess
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import time

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================

# 1. SCRIPT DIRECTORY
SCRIPT_DIR = "/home/users/aadsilva/ic/erad-2026/workbench/PDEBench/pdebench/models/"

# 2. MODEL ROOT
MODEL_ROOT = os.path.expanduser("~/ic/trained-models")

# 3. DATA ROOT
DATA_ROOT = "/home/users/aadsilva/ic/erad-2026/workbench/PDEBench/pdebench/data_download/" 

# Config file name
CONFIG_FILE = "config_Adv.yaml"

# Global Cache for Data Files
DATA_FILE_CACHE = {}

def build_data_cache():
    """Scans DATA_ROOT once and maps filenames to full paths."""
    print(f"\n📦 Building Data File Cache from: {DATA_ROOT}")
    print("   Scanning... (this might take a moment)")
    count = 0
    for root, dirs, files in os.walk(DATA_ROOT):
        for file in files:
            if file.endswith(".hdf5") or file.endswith(".h5"):
                DATA_FILE_CACHE[file] = os.path.join(root, file)
                count += 1
                if count % 20 == 0:
                    print(f"   ---> Found {count} files so far...")
    print(f"✅ Cache built. Total data files found: {count}\n")

def find_data_file(data_filename):
    return DATA_FILE_CACHE.get(data_filename)

def debug_paths():
    print("="*60)
    print("🐛 DEBUGGING PATHS")
    print("="*60)
    print(f"Script Directory : {SCRIPT_DIR}")
    print(f"Model Root       : {MODEL_ROOT}")
    print(f"Data Root        : {DATA_ROOT}")
    
    train_script = os.path.join(SCRIPT_DIR, "train_models_forward.py")
    if not os.path.exists(train_script):
        print(f"❌ CRITICAL: 'train_models_forward.py' not found in {SCRIPT_DIR}")
    else:
        print(f"✅ Found training script: {train_script}")

def parse_filename(filepath):
    filename = os.path.basename(filepath)
    # Remove any possible extension dynamically (.pt or .pickle)
    name_no_ext = filename.replace(".pt", "").replace(".pickle", "")
    parts = name_no_ext.split('_')
    
    if len(parts) < 3: return None

    model_type = parts[0] 
    
    # Extract Parent Folder (Optimization Strategy)
    parent_folder = os.path.basename(os.path.dirname(filepath))
    
    date_idx = -1
    for i, p in enumerate(parts):
        if p.startswith("202") and len(p) == 8 and p.isdigit():
            date_idx = i
            break
            
    if date_idx == -1: return None

    hdf5_base = "_".join(parts[1:date_idx]) 
    hdf5_name = hdf5_base + ".hdf5"
    param_str = parts[date_idx-1] 
    
    return {
        "full_path": filepath,
        "optimization": parent_folder,
        "model_type": model_type,
        "hdf5_base": hdf5_base,
        "hdf5_name": hdf5_name,
        "param_group": f"{parts[1]}_{parts[2]}_{param_str}", 
        "run_id": parts[-1]
    }

def extract_metrics(results):
    """
    Safely extracts metrics from either a dictionary or an array.
    Adapted from analyse_results_forward logic to include FT and Boundary MSEs.
    """
    metrics_res = {
        "MSE": np.nan, "nRMSE": np.nan, "Conservation": np.nan, 
        "MaxError": np.nan, "Boundary": np.nan, "FT_Low": np.nan, 
        "FT_Mid": np.nan, "FT_High": np.nan
    }
    
    if isinstance(results, dict):
        metrics_res["MSE"] = float(results.get("MSE", np.nan))
        metrics_res["nRMSE"] = float(results.get("RMSE", results.get("Normalized MSE", np.nan)))
        metrics_res["Conservation"] = float(results.get("Conservation MSE", np.nan))
        metrics_res["MaxError"] = float(results.get("Max Error", np.nan))
        metrics_res["Boundary"] = float(results.get("MSE Boundary", np.nan))
        metrics_res["FT_Low"] = float(results.get("MSE FT Low", np.nan))
        metrics_res["FT_Mid"] = float(results.get("MSE FT Mid", np.nan))
        metrics_res["FT_High"] = float(results.get("MSE FT High", np.nan))
        
    elif isinstance(results, (list, tuple, np.ndarray)):
        # Common layout: [MSE, nRMSE, Conservation, MaxError, Boundary, [FT_Low, FT_Mid, FT_High]]
        metrics_res["MSE"] = float(results[0]) if len(results) > 0 else np.nan
        metrics_res["nRMSE"] = float(results[1]) if len(results) > 1 else np.nan
        metrics_res["Conservation"] = float(results[2]) if len(results) > 2 else np.nan
        metrics_res["MaxError"] = float(results[3]) if len(results) > 3 else np.nan
        metrics_res["Boundary"] = float(results[4]) if len(results) > 4 else np.nan
        
        # In PINN, the 6th element is often an array itself representing FT components
        if len(results) > 5:
            if isinstance(results[5], (list, tuple, np.ndarray)) and len(results[5]) >= 3:
                metrics_res["FT_Low"] = float(results[5][0])
                metrics_res["FT_Mid"] = float(results[5][1])
                metrics_res["FT_High"] = float(results[5][2])
            else:
                metrics_res["FT_Low"] = float(results[5]) if len(results) > 5 else np.nan
                metrics_res["FT_Mid"] = float(results[6]) if len(results) > 6 else np.nan
                metrics_res["FT_High"] = float(results[7]) if len(results) > 7 else np.nan

    print(f"   📊 nRMSE: {metrics_res['nRMSE']:.5f} | MaxErr: {metrics_res['MaxError']:.5f}")
    return metrics_res

def process_file(meta):
    print(f"----------------------------------------------------------------")
    print(f"📌 PROCESSING: {meta['optimization']} | {meta['model_type']} | {meta['run_id']}")
    print(f"   File: {meta['full_path']}")
    
    # -------------------------------------------------------------
    # FAST PATH: If the file is already a .pickle (e.g. PINN)
    # -------------------------------------------------------------
    if meta['full_path'].endswith('.pickle'):
        print("   🔍 PINN/Evaluated model detected. Reading .pickle directly...")
        try:
            with open(meta['full_path'], "rb") as f:
                results = pickle.load(f)
                return extract_metrics(results)
        except Exception as e:
            print(f"   ❌ Error reading pickle: {e}")
            return None

    # -------------------------------------------------------------
    # INFERENCE PATH: If the file is a .pt (FNO/Unet)
    # -------------------------------------------------------------
    data_path_full = find_data_file(meta["hdf5_name"])
    if not data_path_full:
        print("   ⚠️  HDF5 not in cache, trying .h5 extension...")
        alt_name = meta["hdf5_name"].replace(".hdf5", ".h5")
        data_path_full = find_data_file(alt_name)
        
    if not data_path_full:
        print(f"   ❌ ERROR: Data file {meta['hdf5_name']} NOT FOUND in cache.")
        return None
    
    print(f"   ✅ Data found: {data_path_full}")
    data_dir = os.path.dirname(data_path_full) + "/"

    # 2. Settings
    if meta["model_type"] == "FNO":
        batch_size = 256 
    elif meta["model_type"] == "Unet":
        batch_size = 128
    else:
        print(f"   ❌ Unknown model type: {meta['model_type']}")
        return None

    # 3. Stage Model
    target_pt_name = os.path.join(SCRIPT_DIR, f"{meta['hdf5_base']}_{meta['model_type']}.pt")
    target_pickle_name = os.path.join(SCRIPT_DIR, f"{meta['hdf5_base']}_{meta['model_type']}.pickle")

    try:
        shutil.copyfile(meta["full_path"], target_pt_name)
    except IOError as e:
        print(f"   ❌ ERROR copying model file: {e}")
        return None

    # 4. Command
    actual_filename = os.path.basename(data_path_full)
    cmd = [
        "python", "train_models_forward.py",
        f"+args={CONFIG_FILE}",
        f"++args.model_name={meta['model_type']}",
        "++args.if_training=False",
        "++args.continue_training=False",
        f"++args.filename={actual_filename}", 
        f"++args.data_path={data_dir}",
        f"++args.batch_size={batch_size}",
        "++args.epochs=1"
    ]

    # Force CPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "" 

    metrics_res = None
    start_time = time.time()

    try:
        print(f"   🚀 Running inference subprocess (CPU Mode)...")
        result = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=True, text=True, env=env)
        elapsed = time.time() - start_time

        if result.returncode != 0:
            print(f"   💥 SUBPROCESS FAILED (Exit Code {result.returncode})")
            print(f"   --- STDERR START ---")
            print(result.stderr[-500:]) 
            print(f"   --- STDERR END ---")
        else:
            print(f"   ✅ Finished inference in {elapsed:.2f}s")
            
            if os.path.exists(target_pickle_name):
                with open(target_pickle_name, "rb") as f:
                    results = pickle.load(f)
                    metrics_res = extract_metrics(results)
            else:
                print("   ❌ Picke file was NOT generated.")

    except Exception as e:
        print(f"   💥 Python Exception: {e}")
    finally:
        # Cleanup
        if os.path.exists(target_pt_name): os.remove(target_pt_name)
        if os.path.exists(target_pickle_name): os.remove(target_pickle_name)

    return metrics_res

def main():
    debug_paths()
    build_data_cache()

    print("🔎 Searching for .pt and .pickle files recursively...")
    
    # BUSCA TANTO .PT QUANTO .PICKLE
    pt_files = glob.glob(os.path.join(MODEL_ROOT, "**/*.pt"), recursive=True)
    pickle_files = glob.glob(os.path.join(MODEL_ROOT, "**/*.pickle"), recursive=True)
    
    all_files = pt_files + pickle_files
    all_files.sort() # Força a ordem cronológica/alfabética
    
    print(f"🔎 Found {len(all_files)} files to evaluate ({len(pt_files)} .pt | {len(pickle_files)} .pickle).")
    
    if len(all_files) == 0:
        print("❌ No models found. Exiting.")
        return

    records = []
    
    print("\n========================================")
    print("🚀 STARTING BATCH EVALUATION")
    print("========================================")

    for i, f in enumerate(all_files):
        print(f"\n[{i+1}/{len(all_files)}]")
        meta = parse_filename(f)
        if not meta: 
            print(f"⚠️  Skipping unparseable filename: {f}")
            continue
        
        res = process_file(meta)
        if res:
            row = {
                "Optimization": meta["optimization"],
                "Group": meta["param_group"],
                "Model": meta["model_type"],
                "File": meta["hdf5_name"],
                "Run": meta["run_id"],
                **res
            }
            records.append(row)

    if not records:
        print("\n❌ No results collected. Check errors above.")
        return

    df = pd.DataFrame(records)
    
    print("\n" + "="*60)
    print("📊 GENERATING SUMMARIES")
    print("="*60)

    # Expanded to support Fourier Transform metrics typical for PINN 
    numeric_cols = [
        "nRMSE", "MaxError", "MSE", "Conservation", 
        "Boundary", "FT_Low", "FT_Mid", "FT_High"
    ]

    # Drop strictly null columns if you want cleaner tables, 
    # but keeping them guarantees structural consistency
    # df = df.dropna(axis=1, how='all')
    
    # 1. Detailed Summary
    summary_detailed = df.groupby(["Optimization", "Model", "File"])[numeric_cols].agg(['mean', 'std'])
    summary_detailed.to_csv("Evaluation_Detailed_Per_File.csv")
    print("✅ Saved 'Evaluation_Detailed_Per_File.csv'")

    # 2. High-Level Summary
    summary_opt = df.groupby(["Optimization", "Model"])[numeric_cols].agg(['mean', 'std'])
    summary_opt.to_csv("Evaluation_Summary_Per_Optimization.csv")
    print("✅ Saved 'Evaluation_Summary_Per_Optimization.csv'")
    
    # 3. Raw Logs Dump 
    df.to_csv("Evaluation_Raw_Runs.csv", index=False)
    print("✅ Saved 'Evaluation_Raw_Runs.csv'")
    
    print("\nPreview of Optimization Summary:")
    print(summary_opt)
    
    print("\n🎉 ALL DONE.")

if __name__ == "__main__":
    main()