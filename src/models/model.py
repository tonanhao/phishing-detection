from __future__ import annotations

import torch
import torch.nn as nn


class PhishingRepresentationModel(nn.Module):
    """Hybrid CNN + BiLSTM representation model with reconstruction head."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        repr_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.cnn = nn.Sequential(
            nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        fusion_dim = 128 + hidden_dim * 2
        self.encoder = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, repr_dim),
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(repr_dim, 256),
            nn.ReLU(),
            nn.Linear(256, fusion_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(repr_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor, return_repr: bool = False):
        embedded = self.embedding(x)

        cnn_in = embedded.permute(0, 2, 1)
        cnn_out = self.cnn(cnn_in).squeeze(-1)

        _, (hidden, _) = self.lstm(embedded)
        lstm_out = torch.cat([hidden[-2], hidden[-1]], dim=1)

        fused = torch.cat([cnn_out, lstm_out], dim=1)
        representation = self.encoder(fused)
        reconstruction = self.decoder(representation)
        logits = self.classifier(representation)

        if return_repr:
            return logits, reconstruction, fused, representation
        return logits, reconstruction, fused
