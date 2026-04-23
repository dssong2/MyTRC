from torch import nn
import torch

class ReasoningBlock(nn.Module):
    def __init__(self, d_z: int, d_h: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_z, num_heads, batch_first=True)
        # Feed forward network for processing the output of the attention layer
        self.ff = nn.Sequential(
            nn.Linear(d_z, d_h),
            nn.GELU(),
            nn.Linear(d_h, d_z)
        )
        self.norm1 = nn.LayerNorm(d_z)
        self.norm2 = nn.LayerNorm(d_z)
        
    def forward(self, z: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        seq = torch.stack([z, ctx], dim=1)  # Shape: (batch_size, 2, d_z)
        # Self-attention requires identical arguments for query, key, and value, so we use the same sequence for all three
        # z attending to ctx and ctx attending to z simultaneously
        attn_output, _ = self.attention(seq, seq, seq)  # Self-attention
        z = self.norm1(z + attn_output[:, 0])  # Residual
        z = self.norm2(z + self.ff(z))  # Feed forward with residual
        return z

class ReasoningModule(nn.Module):
    def __init__(self, d_z: int, d_h: int, L: int = 3, num_heads: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList([
            ReasoningBlock(d_z, d_h, num_heads) for _ in range(L)
        ])
        
    def forward(self, z: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            z = block(z, ctx)
        return z