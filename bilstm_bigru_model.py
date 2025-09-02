import torch
import torch.nn as nn

class BiLSTMModel(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, output_dim, pooling, n_layers=2, dropout_rate=0.5, padding_idx=0):
        super(BiLSTMModel, self).__init__()
        embedding_dim = embedding_matrix.shape[1]
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            padding_idx=padding_idx,
            freeze=True,
        )
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate if n_layers > 1 else 0,
        )
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        assert pooling in ["mask_average", "mask_max"]
        self.pooling = pooling

    def forward(self, x):
        embedded = self.embedding(x)
        mask = (x != self.embedding.padding_idx).unsqueeze(-1)
        lstm_output, _ = self.lstm(embedded)
        lstm_output = lstm_output * mask
        if self.pooling == "mask_average":
            summed = lstm_output.sum(dim=1)
            valid_counts = mask.sum(dim=1).clamp(min=1)
            pooled_output = summed / valid_counts
        else:
            pooled_output = torch.max(lstm_output.masked_fill(~mask, float("-inf")), dim=1).values
        output = self.batch_norm(pooled_output)
        output = self.dropout(output)
        output = self.fc(output)
        return output

class BiGRUModel(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, output_dim, pooling, n_layers=2, dropout_rate=0.5, padding_idx=0):
        super(BiGRUModel, self).__init__()
        embedding_dim = embedding_matrix.shape[1]
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            padding_idx=padding_idx,
            freeze=True,
        )
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate if n_layers > 1 else 0,
        )
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        assert pooling in ["mask_average", "mask_max"]
        self.pooling = pooling

    def forward(self, x):
        embedded = self.embedding(x)
        mask = (x != self.embedding.padding_idx).unsqueeze(-1)
        gru_output, _ = self.gru(embedded)
        gru_output = gru_output * mask
        if self.pooling == "mask_average":
            summed = gru_output.sum(dim=1)
            valid_counts = mask.sum(dim=1).clamp(min=1)
            pooled_output = summed / valid_counts
        else:
            pooled_output = torch.max(gru_output.masked_fill(~mask, float("-inf")), dim=1).values
        output = self.batch_norm(pooled_output)
        output = self.dropout(output)
        output = self.fc(output)
        return output

class BiLSTMInference:
    def __init__(self, model_path, embedding_matrix, pooling="mask_average", device="cpu", padding_idx=0):
        self.device = device
        self.model = BiLSTMModel(
            embedding_matrix=embedding_matrix,
            hidden_dim=256,
            output_dim=2,  # or your num classes
            pooling=pooling,
            n_layers=2,
            dropout_rate=0.5,
            padding_idx=padding_idx
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.tokenizer = None

    def set_tokenizer(self, tokenizer_func):
        self.tokenizer = tokenizer_func

    def infer(self, text):
        if self.tokenizer is None:
            raise ValueError("Tokenizer function is not set.")
        indices = self.tokenizer(text)
        indices = indices.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(indices)
            probs = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
        return {"class": predicted_class, "probabilities": probs.cpu().numpy().tolist()}

class BiGRUInference:
    def __init__(self, model_path, embedding_matrix, pooling="mask_average", device="cpu", padding_idx=0):
        self.device = device
        self.model = BiGRUModel(
            embedding_matrix=embedding_matrix,
            hidden_dim=256,
            output_dim=2,
            pooling=pooling,
            n_layers=2,
            dropout_rate=0.5,
            padding_idx=padding_idx,
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.tokenizer = None

    def set_tokenizer(self, tokenizer_func):
        self.tokenizer = tokenizer_func

    def infer(self, text):
        if self.tokenizer is None:
            raise ValueError("Tokenizer function is not set.")
        indices = self.tokenizer(text)
        indices = indices.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(indices)
            probs = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
        return {"class": predicted_class, "probabilities": probs.cpu().numpy().tolist()}
