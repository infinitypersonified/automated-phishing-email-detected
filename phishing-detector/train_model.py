# train_model.py (FIXED + BALANCED)

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm

# -----------------------
# Step 1: Load dataset
# -----------------------
df = pd.read_csv("final_training_dataset.csv")

# 🔥 Fix mixed labels
df["label"] = df["label"].replace({
    "phishing": 1,
    "legitimate": 0
})

df["label"] = df["label"].astype(int)

print("Cleaned dataset distribution:")
print(df["label"].value_counts())

# -----------------------
# 🔥 Step 2: Balance dataset
# -----------------------
phishing_df = df[df["label"] == 1]
legit_df = df[df["label"] == 0]

min_size = min(len(phishing_df), len(legit_df))

phishing_df = phishing_df.sample(min_size, random_state=42)
legit_df = legit_df.sample(min_size, random_state=42)

df = pd.concat([phishing_df, legit_df]).sample(frac=1, random_state=42)

print("\nBalanced dataset:")
print(df["label"].value_counts())

# -----------------------
# 🔥 Step 3: Reduce dataset size (for speed)
# -----------------------
df = df.sample(n=10000, random_state=42)

df = df[['text', 'label']]

# -----------------------
# Step 4: Train/test split
# -----------------------
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['label']
)

print("\nTrain size:", len(train_df), "Test size:", len(test_df))

# -----------------------
# Step 5: Tokenizer
# -----------------------
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

class EmailDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors="pt"
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item

train_dataset = EmailDataset(
    train_df['text'].tolist(),
    train_df['label'].tolist()
)

# -----------------------
# Step 6: DataLoader
# -----------------------
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# -----------------------
# Step 7: Load model
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

model.to(device)

# -----------------------
# Step 8: Optimizer
# -----------------------
optimizer = AdamW(model.parameters(), lr=2e-5)

# -----------------------
# Step 9: Training loop
# -----------------------
epochs = 1

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}")
    model.train()

    loop = tqdm(train_loader, leave=True)

    for batch in loop:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        loop.set_description(f"Loss: {loss.item():.4f}")

# -----------------------
# Step 10: Save model
# -----------------------
model.save_pretrained("fine_tuned_phishing_model", safe_serialization=False)
tokenizer.save_pretrained("fine_tuned_phishing_model")

print("\n✅ Training complete and model saved!")