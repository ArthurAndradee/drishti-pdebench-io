from __future__ import annotations
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

def main():
    # 1. Get all pickle files
    files = list(Path().glob("*.pickle"))
    files.sort()
    
    if not files:
        print("No .pickle files found in the current directory.")
        return

    print(f"Found {len(files)} pickle files: {[f.name for f in files]}")

    # 2. Define Metric Names (Standard from PDEBench)
    metric_names = [
        "MSE",
        "Normalized MSE",
        "Conservation MSE",
        "Max Error",
        "MSE Boundary",
        "MSE FT Low",
        "MSE FT Mid",
        "MSE FT High"
    ]

    # 3. Load Data
    records = []
    
    for fl in files:
        # Load the pickle data
        with open(fl, "rb") as f:
            try:
                # The pickle contains a dictionary of metrics
                data = pickle.load(f)
            except Exception as e:
                print(f"Error loading {fl}: {e}")
                continue

        # Extract metrics into a list
        # Note: The pickle structure from train.py returns a dictionary.
        # We need to extract values in the specific order or by key.
        # Based on typical PDEBench metrics output, it usually returns a dict.
        # If it returns a list/tuple (as implied by the original code), we use it directly.
        
        row_metrics = []
        if isinstance(data, dict):
            # Map dictionary keys to our list if it's a dict
            row_metrics = [
                data.get("MSE", np.nan),
                data.get("RMSE", np.nan), # Normalized MSE usually
                0.0, # Conservation placeholder if missing
                data.get("Max Error", np.nan),
                0.0, # Boundary placeholder
                0.0, 0.0, 0.0 # FT placeholders
            ]
        elif isinstance(data, (list, tuple, np.ndarray)):
            # If it's a flat list (original PDEBench behavior)
            row_metrics = list(data)
            
        # Pad with NaNs if metrics are missing
        if len(row_metrics) < len(metric_names):
            row_metrics += [np.nan] * (len(metric_names) - len(row_metrics))

        # 4. Parse Filename for Indexing
        # Filename format: 1D_Advection_Sols_beta1.0_FNO.pickle
        name_parts = fl.stem.split('_')
        
        # Default fallbacks
        pde_type = "Unknown"
        param = "Unknown"
        model = "Unknown"

        # Logic for your specific file: 1D_Advection_Sols_beta1.0_FNO
        if "Advection" in fl.name:
            pde_type = "1D_Advection"
            # Extract beta value
            beta_match = re.search(r"beta([\d\.]+)", fl.name)
            param = f"beta={beta_match.group(1)}" if beta_match else "beta=?"
            # Extract model (last part before extension)
            model = name_parts[-1]
        else:
            # Fallback for other files
            pde_type = name_parts[0] if len(name_parts) > 0 else "Unknown"
            model = name_parts[-1] if len(name_parts) > 0 else "Unknown"

        records.append([pde_type, param, model] + row_metrics)

    # 5. Create DataFrame
    columns = ["PDE", "Param", "Model"] + metric_names
    df = pd.DataFrame(records, columns=columns)
    
    # Set MultiIndex
    df.set_index(["PDE", "Param", "Model"], inplace=True)

    # 6. Save to CSV
    print("\nGenerated Results.csv:")
    print(df)
    df.to_csv("Results.csv")

    # 7. Plotting (Simplified)
    # Only plot if we have data
    if not df.empty:
        try:
            # Reset index for easier plotting
            plot_df = df.reset_index()
            
            pdes = plot_df["PDE"].unique()
            models = plot_df["Model"].unique()
            
            x = np.arange(len(pdes))
            width = 0.8 / len(models)

            fig, ax = plt.subplots(figsize=(10, 6))

            for i, model in enumerate(models):
                model_data = plot_df[plot_df["Model"] == model]
                # Match PDEs to ensure alignment
                heights = []
                for pde in pdes:
                    val = model_data[model_data["PDE"] == pde]["MSE"].values
                    heights.append(val[0] if len(val) > 0 else 0)
                
                position = x - 0.4 + width * i + (width/2)
                ax.bar(position, heights, width, label=model)

            ax.set_xticks(x)
            ax.set_xticklabels(pdes, fontsize=12, rotation=45)
            ax.set_ylabel("MSE", fontsize=14)
            ax.set_yscale("log")
            ax.legend()
            ax.set_title("Model Comparison")
            
            plt.tight_layout()
            plt.savefig("Results.png")
            print("\nSaved plot to Results.png")
            
        except Exception as e:
            print(f"Skipping plot generation due to error: {e}")

if __name__ == "__main__":
    main()