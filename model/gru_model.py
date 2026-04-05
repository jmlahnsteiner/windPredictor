"""
model/gru_model.py — PyTorch GRU for sailing condition forecasting.

Architecture:
  Sequence input : (B, T=24, seq_features=7)  — hourly station readings
  Context input  : (B, context_features=12)   — NWP + time context

  GRU(seq_features → hidden_size, num_layers=2, dropout)
  → last hidden state (B, hidden_size)
  → concat context   (B, hidden_size + context_features)
  → Linear → ReLU → Dropout → Linear → Sigmoid
  → (B, 1)  probability
"""

import torch
import torch.nn as nn

from model.features_sequence import CONTEXT_FEATURES, SEQ_FEATURES


class SailingGRU(nn.Module):
    def __init__(
        self,
        seq_features: int = SEQ_FEATURES,
        context_features: int = CONTEXT_FEATURES,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=seq_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        combined = hidden_size + context_features
        self.head = nn.Sequential(
            nn.Linear(combined, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, seq: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """
        seq : (B, T, seq_features)
        ctx : (B, context_features)
        returns (B, 1) probabilities
        """
        _, hidden = self.gru(seq)       # hidden: (num_layers, B, hidden_size)
        last_hidden = hidden[-1]        # (B, hidden_size)
        combined = torch.cat([last_hidden, ctx], dim=1)
        return self.head(combined)
