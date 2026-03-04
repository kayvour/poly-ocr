import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_plots(csv_path):
    plt.style.use("ggplot")
    plt.figure(figsize=(6,4))

    df = pd.read_csv(csv_path)

    os.makedirs("results/plots", exist_ok=True)

    # Average metrics per engine
    grouped = df.groupby("engine").mean(numeric_only=True)

    # CER Plot
    grouped["cer"].plot(kind="bar")
    plt.title("Average CER per OCR Engine")
    plt.ylabel("CER")
    plt.savefig("results/plots/cer_comparison.png")
    plt.clf()

    # WER Plot
    grouped["wer"].plot(kind="bar")
    plt.title("Average WER per OCR Engine")
    plt.ylabel("WER")
    plt.savefig("results/plots/wer_comparison.png")
    plt.clf()

    # Speed Plot
    grouped["time"].plot(kind="bar")
    plt.title("Average Inference Time")
    plt.ylabel("Seconds")
    plt.savefig("results/plots/speed_comparison.png")
    plt.clf()

    print("Plots saved in results/plots/")
