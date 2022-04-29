# model.py
# Define the LSTM model

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs


class LSTM_net(nn.Module):
    def __init__(self, embedding, embed_size, vocab_size, hidden_dim, num_layers, dropout=0):
        super(LSTM_net, self).__init__()

        if embedding is not None:
            self.embedding = nn.Embedding(vocab_size, embed_size)
            self.embedding.weight = nn.Parameter(embedding)
            self.embedding.weight.requires_grad = False
            self.pretrain = 1
        else:
            self.embedding = nn.Embedding(vocab_size, embed_size, max_norm=1)
            self.embedding.weight.requires_grad = True
            self.pretrain = 0

        self.embed_size = embed_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        if num_layers == 1:
            self.lstm = nn.LSTM(embed_size,
                                hidden_dim,
                                num_layers=num_layers,
                                batch_first=True,
                                dropout=0)
        else:
            self.lstm = nn.LSTM(embed_size,
                                hidden_dim,
                                num_layers=num_layers,
                                batch_first=True,
                                dropout=dropout)
        self.attention = Attention(hidden_dim)
        self.classifier = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(hidden_dim, 5),
                                        nn.Softmax(dim=1))

        # nn.init.orthogonal_(self.lstm.weight_ih_l0)
        # nn.init.orthogonal_(self.lstm.weight_hh_l0)
        # nn.init.zeros_(self.lstm.bias_ih_l0)
        # nn.init.zeros_(self.lstm.bias_hh_l0)

    def forward(self, inputs):
        inputs = self.embedding(inputs)
        x, _ = self.lstm(inputs, None)
        # x = [batch, seq_len, hidden_size]
        x = self.attention(x)
        # x = x[:, -1, :]
        x = self.classifier(x)
        return x
