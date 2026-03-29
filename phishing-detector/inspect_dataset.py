import pandas as pd

df = pd.read_csv("final_training_dataset.csv")

print(df.head())          # See first 5 rows
print(df["label"].value_counts())  # Check balance: phishing vs legit
