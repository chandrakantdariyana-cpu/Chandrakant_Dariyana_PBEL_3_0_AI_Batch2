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

        plt.style.use("dark_background")

        plt.figure(figsize=(12, 8))

        attack_count = attack_count.sort_values(ascending=True)

        bars = plt.barh(
            attack_count.index,
            attack_count.values,
            color="royalblue",
            edgecolor="white"
        )

        plt.xscale("log")

        plt.title(
            "CICIDS2017 Attack Distribution",
            fontsize=20,
            fontweight="bold",
            pad=15
        )

        plt.xlabel(
            "Number of Records (Log Scale)",
            fontsize=14
        )

        plt.ylabel(
            "Attack Category",
            fontsize=14
        )

        plt.grid(
            axis="x",
            linestyle="--",
            alpha=0.4
        )

        # Display values on bars
        for bar in bars:
            width = bar.get_width()

            plt.text(
                width * 0.95,  # Bar ke andar
                bar.get_y() + bar.get_height() / 2,
                f"{int(width):,}",
                va="center",
                ha="right",  # Right align
                fontsize=9,
                color="white",
                fontweight="bold"
            )

        plt.tight_layout()

        save_path = IMAGE_DIR / "Attack_Distribution_Horizontal.png"

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()

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