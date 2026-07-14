import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




df = pd.read_csv("../CICIDS2017_Final.csv")
print(df.columns.tolist())


print(df.shape)
print(df.info())
print(df.head())

print(df.isnull().sum())

print(df.duplicated().sum())
df = df.drop_duplicates()

numeric_df = df.select_dtypes(include=[np.number])
print(np.isinf(numeric_df).sum())

df.replace([np.inf, -np.inf], np.nan, inplace=True)

print(df.isnull().sum())
df = df.dropna()


print(df.shape)
print(df.isnull().sum())
print(df.duplicated().sum())


print("\nFinal Shape :", df.shape)

print("\nRemaining Missing Values")
print(df.isnull().sum().sum())

print("\nDuplicate Rows :", df.duplicated().sum())
df.to_csv("CICIDS2017_Clean.csv", index=False)
print("\nDataset cleaned successfully!")

