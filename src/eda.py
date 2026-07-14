"""
===========================================================
Project : AI-Based Cyber Threat Detection Framework
File    : attack_distribution.py
Author  : Chandrakant
Purpose : Visualize attack label distribution in the
          cleaned CICIDS2017 dataset.
===========================================================
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "CICIDS2017_Clean.csv"

IMAGE_DIR = BASE_DIR / "images"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Attack Distribution Function
# ==========================================================

def plot_attack_distribution():

    try:

        print("=" * 60)
        print("Loading cleaned dataset...")

        df = pd.read_csv(DATA_PATH)

        print("Dataset loaded successfully.")
        print(f"Dataset Shape : {df.shape}")

        # Remove unwanted spaces
        df.columns = df.columns.str.strip()

        # Count attack labels
        attack_count = df["Label"].value_counts()

        print("\nGenerating attack distribution graph...")

        plt.figure(figsize=(14, 7))

        attack_count.plot(
            kind="bar",
            color="steelblue",
            edgecolor="black"
        )

        plt.title(
            "Attack Distribution in CICIDS2017",
            fontsize=18,
            fontweight="bold"
        )

        plt.xlabel("Attack Type", fontsize=14)

        plt.ylabel("Number of Records", fontsize=14)

        plt.xticks(rotation=45, ha="right")

        plt.grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()

        save_path = IMAGE_DIR / "Attack_Distribution.png"

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(f"Graph saved to : {save_path}")

        plt.show()

        plt.close()

        print("Program completed successfully.")
        print("=" * 60)

    except FileNotFoundError:

        print("ERROR : Clean dataset not found.")
        print(f"Expected Location : {DATA_PATH}")

    except Exception as error:

        print(f"Unexpected Error : {error}")


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":
    plot_attack_distribution()