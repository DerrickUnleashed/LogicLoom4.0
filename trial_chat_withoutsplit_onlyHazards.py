import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load dataset
train_df = pd.read_csv("Hazards_LABELLED_TRAIN_onlyHazards.csv")
test_df = pd.read_csv("Hazards_UNLABELLED_TEST.csv")

# Encode target labels
label_enc = LabelEncoder()
train_df['hazard-type-enc'] = label_enc.fit_transform(train_df['hazard-type'])

# Tokenizer (using BERT)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Dataset class
class HazardDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = str(self.df.iloc[idx]['text'])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        if not self.is_test:
            item['labels'] = torch.tensor(self.df.iloc[idx]['hazard-type-enc'], dtype=torch.long)
        return item

# Data preparation
train_dataset = HazardDataset(train_df, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

test_dataset = HazardDataset(test_df, tokenizer, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Model definition
class HazardClassifier(nn.Module):
    def __init__(self, n_classes):
        super(HazardClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        output = self.drop(pooled_output)
        return self.out(output)

model = HazardClassifier(n_classes=len(label_enc.classes_))
model = model.to(device)

# Optimizer and loss
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

# Training loop on full training set
epochs = 4

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_loss:.4f}")

# Prediction on test set
model.eval()
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        outputs = model(input_ids, attention_mask)
        _, predicted = torch.max(outputs, dim=1)
        test_preds.extend(predicted.cpu().numpy())

# Map back to labels
pred_labels = label_enc.inverse_transform(test_preds)

# Prepare submission
submission = pd.DataFrame({
    'ID': test_df['ID'],
    'hazard': pred_labels
})

submission.to_csv('submission4.csv', index=False)
print("Submission file 'submission.csv' created successfully.")
