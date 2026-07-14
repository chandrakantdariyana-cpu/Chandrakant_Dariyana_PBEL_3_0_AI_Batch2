import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../CICIDS2017_Clean.csv")

# Numeric columns only
numeric_df = df.select_dtypes(include="number")

# Correlation
corr = numeric_df.corr()

# Select first 20 features
top_corr = corr.iloc[:20, :20]

plt.figure(figsize=(14,10))

plt.imshow(top_corr, cmap="coolwarm", interpolation="nearest")

plt.colorbar(label="Correlation")

plt.xticks(range(len(top_corr.columns)),
           top_corr.columns,
           rotation=90,
           fontsize=8)

plt.yticks(range(len(top_corr.columns)),
           top_corr.columns,
           fontsize=8)

plt.title("Top 20 Feature Correlation Heatmap")

plt.tight_layout()

plt.show()