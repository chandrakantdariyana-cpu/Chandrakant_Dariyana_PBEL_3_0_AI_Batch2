import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("../CICIDS2017_Clean.csv")

# Remove spaces from column names
df.columns = df.columns.str.strip()

# Count labels
attack_count = df["Label"].value_counts()

# Plot
plt.figure(figsize=(14,7))
attack_count.plot(kind="bar", color="steelblue", edgecolor="black")

plt.title("Attack Distribution in CICIDS2017", fontsize=18)
plt.xlabel("Attack Type", fontsize=14)
plt.ylabel("Number of Records", fontsize=14)

plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig("../images/Attack_Distribution.png", dpi=300)

plt.show()