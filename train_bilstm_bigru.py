import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from bilstm_bigru_model import BiLSTMModel, BiGRUModel
from tokenizer import simple_tokenizer

# Load vocab and embedding matrix
with open("vocab.json") as f:
    vocab = json.load(f)

embedding_matrix = np.load("embedding_matrix.npy")
padding_idx = vocab["<PAD>"]

class MyDataset(Dataset):
    def __init__(self, data_file):
        self.data = []
        with open(data_file) as fin:
            for line in fin:
                item = json.loads(line)
                self.data.append(item)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        label = self.data[idx]["label"]
        indices = simple_tokenizer(text)
        return indices, label

def collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    indices = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch]).long()
    indices_padded = pad_sequence(indices, batch_first=True, padding_value=padding_idx)
    return indices_padded, labels

# Parameters
hidden_dim = 256
output_dim = 2
pooling = "mask_average"
n_layers = 2
dropout_rate = 0.5
num_epochs = 10
batch_size = 16
device = "cuda" if torch.cuda.is_available() else "cpu"

train_set = MyDataset("train.jsonl")
val_set = MyDataset("valid.jsonl")
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

def train_and_validate(model, model_name, filename):
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
        total_loss = 0
        correct = 0
        n = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                n += labels.size(0)
        val_loss = total_loss / n
        val_acc = correct / n
        print(f"[{model_name}] Epoch {epoch+1}: Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), filename)
            print(f"Best {model_name} model saved!")

# BiLSTM training
bilstm_model = BiLSTMModel(
    embedding_matrix, hidden_dim, output_dim, pooling, n_layers, dropout_rate, padding_idx=padding_idx
).to(device)

train_and_validate(bilstm_model, "BiLSTM", "your_bilstm_model.pth")

# BiGRU training
bigru_model = BiGRUModel(
    embedding_matrix, hidden_dim, output_dim, pooling, n_layers, dropout_rate, padding_idx=padding_idx
).to(device)

train_and_validate(bigru_model, "BiGRU", "your_bigru_model.pth")

print("Training complete. Best models saved as your_bilstm_model.pth and your_bigru_model.pth.")
