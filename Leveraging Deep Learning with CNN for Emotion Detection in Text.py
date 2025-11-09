import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import RobertaTokenizer, RobertaModel
import gensim.downloader as api


class TextDataset(Dataset):
    def __init__(self, texts, labels, embeddings, embedding_type):
        self.texts = texts
        self.labels = labels
        self.embeddings = embeddings
        self.embedding_type = embedding_type
        if embedding_type == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text, label = self.texts[idx], self.labels[idx]
        if self.embedding_type == 'roberta':
            tokens = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=50,
                return_tensors="pt"
            )
            return (
                tokens['input_ids'].squeeze(0),
                tokens['attention_mask'].squeeze(0),
                torch.tensor(label, dtype=torch.long)
            )
        else:
            words = text.split()
            vecs = [self.embeddings[w] for w in words if w in self.embeddings]
            if len(vecs) == 0:
                vecs = [np.zeros(300)]
            mat = np.zeros((50, 300))
            for i, v in enumerate(vecs[:50]):
                mat[i] = v
            return torch.tensor(mat, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class CNNModel(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, embedding_dim)
        x = x.permute(0, 2, 1)  # (batch, embedding_dim, seq_len)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.adaptive_max_pool1d(x, 1).squeeze(2)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class RobertaCNN(nn.Module):
    def __init__(self, num_classes):
        super(RobertaCNN, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.conv1 = nn.Conv1d(768, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state  # (batch, seq_len, hidden=768)
        x = x.permute(0, 2, 1)        # (batch, hidden, seq_len)
        x = F.relu(self.conv1(x))
        x = F.adaptive_max_pool1d(x, 1).squeeze(2)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# Dummy data (5-class toy example)
texts = [
    "I am so happy today",
    "This is terrible and makes me angry",
    "I feel sad",
    "That was scary",
    "Wow what a surprise"
] * 200
labels = [0, 1, 2, 3, 4] * 200

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Load embeddings
fasttext_model = api.load("fasttext-wiki-news-subwords-300")
glove_model = api.load("glove-wiki-gigaword-300")

# Datasets
fasttext_train = TextDataset(X_train, y_train, fasttext_model, 'fasttext')
fasttext_test = TextDataset(X_test, y_test, fasttext_model, 'fasttext')

glove_train = TextDataset(X_train, y_train, glove_model, 'glove')
glove_test = TextDataset(X_test, y_test, glove_model, 'glove')

roberta_train = TextDataset(X_train, y_train, None, 'roberta')
roberta_test = TextDataset(X_test, y_test, None, 'roberta')

# Loaders
fasttext_loader_train = DataLoader(fasttext_train, batch_size=32, shuffle=True)
fasttext_loader_test = DataLoader(fasttext_test, batch_size=32)

glove_loader_train = DataLoader(glove_train, batch_size=32, shuffle=True)
glove_loader_test = DataLoader(glove_test, batch_size=32)

roberta_loader_train = DataLoader(roberta_train, batch_size=8, shuffle=True)
roberta_loader_test = DataLoader(roberta_test, batch_size=8)


def train_model(model, train_loader, test_loader, embedding_type, epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            if embedding_type == 'roberta':
                input_ids, attn_mask, labels_b = batch
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)
                labels_b = labels_b.to(device)
                outputs = model(input_ids, attn_mask)
            else:
                inputs, labels_b = batch
                inputs = inputs.to(device)
                labels_b = labels_b.to(device)
                outputs = model(inputs)

            loss = criterion(outputs, labels_b)
            loss.backward()
            optimizer.step()

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            if embedding_type == 'roberta':
                input_ids, attn_mask, labels_b = batch
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)
                labels_b = labels_b.to(device)
                outputs = model(input_ids, attn_mask)
            else:
                inputs, labels_b = batch
                inputs = inputs.to(device)
                labels_b = labels_b.to(device)
                outputs = model(inputs)

            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels_b.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    return acc, precision, recall, f1


# Models
fasttext_cnn = CNNModel(300, 5)
glove_cnn = CNNModel(300, 5)
roberta_cnn = RobertaCNN(5)

# Train & evaluate
fasttext_results = train_model(
    fasttext_cnn, fasttext_loader_train, fasttext_loader_test, 'fasttext'
)
glove_results = train_model(
    glove_cnn, glove_loader_train, glove_loader_test, 'glove'
)
roberta_results = train_model(
    roberta_cnn, roberta_loader_train, roberta_loader_test, 'roberta'
)

print("FastText+CNN:", fasttext_results)
print("GloVe+CNN:", glove_results)
print("RoBERTa+CNN:", roberta_results)
