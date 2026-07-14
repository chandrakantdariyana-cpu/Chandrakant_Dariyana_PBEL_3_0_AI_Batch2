import pandas as pd

# Load cleaned dataset
df = pd.read_csv("../CICIDS2017_Clean.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Display all column names
print("=" * 60)
print("Columns in Dataset")
print("=" * 60)
print(df.columns.tolist())

# Display attack distribution
print("\n" + "=" * 60)
print("Attack Distribution")
print("=" * 60)
print(df["Label"].value_counts())