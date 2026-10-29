import pandas as pd

# Load only the Label column
df = pd.read_csv(
    "C:/Users/pauln/OneDrive/Documents/dos attack/combinenew.csv",
    usecols=[" Label"],
    low_memory=False
)

print("Total Rows:", len(df))
print("\nUnique Labels:")
print(df[" Label"].value_counts())