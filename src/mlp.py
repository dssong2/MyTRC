from torch import nn
import torch

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    # Called via MLP(...)(x) by default, but can also be called directly as MLP.forward(x)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)