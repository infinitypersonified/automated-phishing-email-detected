import pandas as pd

df = pd.read_csv("final_training_dataset.csv")

# Map string labels to 0/1
df["label"] = df["label"].replace({
    "phishing": 1,
    "legitimate": 0
})

# Optional: make sure everything is int
df["label"] = df["label"].astype(int)

# Check counts
print(df["label"].value_counts())

# Save clean dataset
df.to_csv("final_training_dataset_clean.csv", index=False)
print("Cleaned dataset saved!")