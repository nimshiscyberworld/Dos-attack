import pandas as pd
from sklearn.model_selection import train_test_split

print("=" * 60)
print("Loading Cleaned Dataset...")
print("=" * 60)

# Load cleaned dataset
df = pd.read_csv("cleaned_dos_dataset.csv")

print("Original Shape:", df.shape)

# Create a stratified sample of 250,000 records
sampled_df, _ = train_test_split(
    df,
    train_size=250000,
    stratify=df["Label"],
    random_state=42
)

print("\nSampled Shape:", sampled_df.shape)

print("\nClass Distribution:")
print(sampled_df["Label"].value_counts())

# Save sampled dataset
sampled_df.to_csv("sampled_dos_dataset.csv", index=False)

print("\nSampled dataset saved successfully!")