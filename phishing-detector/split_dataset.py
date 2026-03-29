from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("final_training_dataset_clean.csv")

train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)

print("Train and test sets saved!")
print("Train size:", len(train), "Test size:", len(test))