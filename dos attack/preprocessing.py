import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

print("=" * 60)
print("Loading CICIDS2017 Dataset...")
print("=" * 60)

# Dataset path
file_path = r"C:\Users\pauln\OneDrive\Documents\dos attack\dataset\combinenew.csv"

# Load dataset
df = pd.read_csv(file_path, low_memory=False)

print("Original Dataset Shape:", df.shape)

# Remove spaces from column names
df.columns = df.columns.str.strip()

# Keep only DoS attack classes
required_classes = [
    "BENIGN",
    "DoS Hulk",
    "DoS GoldenEye",
    "DoS slowloris",
    "DoS Slowhttptest",
    "DDoS"
]

df = df[df["Label"].isin(required_classes)]

print("Filtered Dataset Shape:", df.shape)

print("\nClass Distribution:")
print(df["Label"].value_counts())

# Replace infinity values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remove missing values
df.dropna(inplace=True)

print("\nDataset Shape after Cleaning:", df.shape)

# Encode labels
encoder = LabelEncoder()
df["Label"] = encoder.fit_transform(df["Label"])

print("\nEncoded Classes:")
for i, cls in enumerate(encoder.classes_):
    print(f"{cls} --> {i}")

# Save cleaned dataset
df.to_csv("cleaned_dos_dataset.csv", index=False)

print("\nCleaned dataset saved as cleaned_dos_dataset.csv")
print("=" * 60)
print("Preprocessing Completed Successfully")
print("=" * 60)