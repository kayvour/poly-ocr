import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_plots(csv_path):
    plt.style.use("ggplot")
    plt.figure(figsize=(6,4))

    df = pd.read_csv(csv_path)

    os.makedirs("results/plots", exist_ok=True)

    # Clean Dataset Average Metrics
    df_clean = df[df["corruption"] == "clean"] if "corruption" in df.columns else df
    grouped = df_clean.groupby("engine").mean(numeric_only=True)

    # Standard bar comparisons
    metrics_to_plot = ["cer", "wer", "exact_match", "bleu", "rouge", "time", "memory_mb"]
    for metric in metrics_to_plot:
        if metric in grouped.columns:
            grouped[metric].plot(kind="bar")
            plt.title(f"Average {metric} per OCR Engine (Clean)")
            plt.ylabel(metric)
            plt.tight_layout()
            plt.savefig(f"results/plots/{metric}_comparison.png")
            plt.clf()

    # Speed vs Accuracy (1 - CER)
    if "time" in grouped.columns and "cer" in grouped.columns:
        plt.figure(figsize=(8,6))
        for idx, row in grouped.iterrows():
            plt.scatter(row["time"], 1.0 - row["cer"], s=100, label=idx)
            plt.text(row["time"], 1.0 - row["cer"], idx, fontsize=12)
        plt.title("Speed vs Accuracy (Clean dataset)")
        plt.xlabel("Average Inference Time (s)")
        plt.ylabel("Accuracy (1 - CER)")
        plt.grid(True)
        plt.savefig("results/plots/speed_vs_accuracy.png")
        plt.clf()
        
    # Robustness degradation curve (CER across corruptions)
    if "corruption" in df.columns and len(df["corruption"].unique()) > 1:
        robustness_grouped = df.groupby(["engine", "corruption"])["cer"].mean().unstack("engine")
        robustness_grouped.plot(kind="bar", figsize=(10, 6))
        plt.title("Robustness: CER across different corruptions")
        plt.ylabel("CER (Lower is better)")
        plt.xlabel("Corruption Type")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("results/plots/robustness_degradation.png")
        plt.clf()

    print("Plots saved in results/plots/")
