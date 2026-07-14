import pandas as pd
import numpy as np

# Load cleaned dataset
df = pd.read_csv("../CICIDS2017_Clean.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print("Dataset Shape:", df.shape)


X = df.drop("Label", axis=1)
y = df["Label"]

print("Features:", X.shape)
print("Target:", y.shape)


from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0)

X_selected = selector.fit_transform(X)

selected_columns = X.columns[selector.get_support()]

print("Original Features :", X.shape[1])
print("Remaining Features:", len(selected_columns))



selected_df = pd.DataFrame(X_selected, columns=selected_columns)

selected_df["Label"] = y.values

selected_df.to_csv("CICIDS2017_FeatureSelected.csv", index=False)

print("Feature Selected Dataset Saved Successfully!")
print(selected_df.shape)