import pandas as pd

files = [
    "CEAS_08.csv",
    "emails.csv",
    "email_phishing_data.csv",
    "Enron.csv",
    "Ling.csv",
    "Nazario.csv",
    "Nigerian_Fraud.csv",
    "phishing_email.csv",
    "SpamAssasin.csv"
]

dfs = []

for file in files:
    df = pd.read_csv(f"dataset/{file}")

    # Try to detect text column
    if "text" in df.columns:
        text_col = "text"
    elif "message" in df.columns:
        text_col = "message"
    elif "body" in df.columns:
        text_col = "body"
    else:
        continue  # skip unknown format

    # Try to detect label column
    if "label" in df.columns:
        label_col = "label"
    elif "spam" in df.columns:
        label_col = "spam"
    elif "phishing" in df.columns:
        label_col = "phishing"
    else:
        continue

    # Keep only needed columns
    df = df[[text_col, label_col]]

    # Rename to standard
    df.columns = ["text", "label"]

    dfs.append(df)

# Merge clean datasets
data = pd.concat(dfs, ignore_index=True)

# Clean
data = data.drop_duplicates(subset="text")
data = data.dropna(subset=["text", "label"])

# Save
data.to_csv("final_training_dataset.csv", index=False)

print("Done. Clean dataset ready.")