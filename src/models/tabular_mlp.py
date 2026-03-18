from __future__ import annotations

import torch
import torch.nn as nn


class TabularMLPBaseline(nn.Module):
    """MLP baseline for tabular phishing datasets with representation bottleneck."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        repr_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, repr_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(repr_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor, return_repr: bool = False):
        fused = x
        representation = self.encoder(fused)
        reconstruction = self.decoder(representation)
        logits = self.classifier(representation)

        if return_repr:
            return logits, reconstruction, fused, representation
        return logits, reconstruction, fused
