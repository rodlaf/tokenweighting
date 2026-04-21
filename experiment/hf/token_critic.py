from __future__ import annotations

import torch
import torch.nn as nn


class TokenValueHead(nn.Module):
    """Small MLP that predicts a per-token scalar value from hidden states."""

    def __init__(self, hidden_size: int = 1536, critic_hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, critic_hidden),
            nn.ReLU(),
            nn.Linear(critic_hidden, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
        Returns:
            values: (batch, seq_len)
        """
        return self.net(hidden_states.float()).squeeze(-1)
