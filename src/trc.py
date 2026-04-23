import numpy as np
from torch import nn
import torch.nn.functional as F
import torch
from src.mlp import MLP
from src.reasoning_module import ReasoningModule
from src.van_der_pol import VanDerPol

class TRC(nn.Module):
    def __init__(self,
        d_x: int,
        d_u: int,
        T: int,
        K: int = 3,
        n: int = 4,
        d_z: int = 256,
        d_h: int = 512,
        L: int = 3,
        num_heads: int = 8,
        u_min: float = -1.0,
        u_max: float = 1.0
    ):
        
        super().__init__()
        
        self.d_x, self.d_u = d_x, d_u
        self.T, self.K, self.n = T, K, n
        self.d_z, self.d_h = d_z, d_h
        self.L = L
        self.num_heads = num_heads
        self.u_min = u_min
        self.u_max = u_max
        
        self.state_encoder = MLP(2 * d_x + 1, d_z, d_h)
        self.error_embed = MLP(d_x, d_z, d_h)
        self.ctrl_embed = nn.Linear(T * d_u, d_z) # Input dimension is T * d_u for the entire control sequence, output dimension is d_z
        
        self.reasoning = ReasoningModule(d_z, d_h, L, num_heads)
        self.initial_decoder = MLP(d_z, self.T*d_u, d_h) # What
        self.residual_decoder = MLP(d_z + self.T*d_u, self.T*d_u, d_h) # What
        
        self.H_init = nn.Parameter(torch.randn(1, d_z) * 0.02)  # Learnable initial hidden state for the reasoning module
        self.L_init = nn.Parameter(torch.randn(1, d_z) * 0.02)  # Learnable initial state for the reasoning module
        self.H_proj = nn.Linear(d_z, d_z)  # Projection layer for the hidden state in the reasoning module
        self.L_proj = nn.Linear(d_z, d_z)  # Projection layer for the state in the reasoning module
        
        self.van_der_pol = VanDerPol(mu=1.0, dt=0.05)

    def _init_latents(self, z0: torch.Tensor):
        # Initialize the hidden state and state for the reasoning module using learnable parameters
        B = z0.shape[0]
        zH = self.H_init.expand(B, -1) + self.H_proj(z0)
        zL = self.L_init.expand(B, -1) + self.L_proj(z0)
        return zH, zL
    
    def forward(self, x0, x_target, dynamics_fn, return_all_iters=False):
        B = x0.shape[0]
        device = x0.device
        
        t_rem = torch.ones(B, 1, device=device) * self.T  # Remaining time steps, shape (B, 1)
        z0 = self.state_encoder(
            torch.cat([x0, x_target, t_rem], dim=-1)
        )
        zH, zL = self._init_latents(z0)
        
        u = self.initial_decoder(z0).view(B, self.T, self.d_u)  # Initial control sequence, shape (B, T, d_u)
        u = u.clamp(self.u_min, self.u_max)  # Clamp initial control to valid range
        all_u = [u] if return_all_iters else None
        
        for k in range(self.K):
            x_T = dynamics_fn(x0, u)
            e = x_T - x_target
            z_ctx = z0 + self.error_embed(e) + self.ctrl_embed(u.flatten(1))
            
            for i in range(self.n):
                zL = self.reasoning(zL, zH + z_ctx)
            
            zH = self.reasoning(zH, zL)
            delta_u = self.residual_decoder(
                torch.cat([zH, u.flatten(1)], dim=-1)
            ).view(B, self.T, self.d_u)
            
            u = (u + delta_u).clamp(self.u_min, self.u_max)  # Update control and clamp to valid range
            if return_all_iters:
                all_u.append(u)
            
        if return_all_iters:
            return all_u  # List of control sequences from each iteration
        else:
            return u  # Final control sequence
        
    def cost(
        self,
        traj: torch.Tensor,  # Shape (B, T+1, d_x), get using van_der_pol_traj(x0, u_seq) -> traj
        u_seq: torch.Tensor,  # Shape (B, T, d_u)
        Q: torch.Tensor,  # Shape (d_x, d_x)
        R: float,  # Shape (d_u, d_u) or scalar
        Qf: torch.Tensor  # Shape (d_x, d_x)
    ) -> torch.Tensor:
        
        x = traj[:, :-1]
        x_T = traj[:, -1]
        
        xQx = (x @ Q) * x
        running = xQx.sum(-1).sum(-1) + R * u_seq.pow(2).sum(-1).sum(-1)
        terminal = ((x_T @ Qf) * x_T).sum(-1)
        
        return running + terminal
    
    def loss(
        self,
        all_u: list,
        u_star: torch.Tensor,
        x0: torch.Tensor,
        Q: torch.Tensor,
        R: float,
        Qf: torch.Tensor,
        lam: float = 1.0
    ) -> tuple[torch.Tensor, dict]:
        
        K = len(all_u) - 1
        
        final_loss = F.mse_loss(all_u[-1], u_star)
        
        costs = [self.cost(self.van_der_pol.traj(x0, u), u, Q, R, Qf) for u in all_u]
        
        J0 = costs[0].detach() + 1e-8
        
        improvement = torch.tensor(0.0, device=u_star.device)
        for k in range(K):
            J_prev = costs[k] / J0
            J_curr = costs[k+1] / J0
            improvement += (J_prev - J_curr).mean()
            
        improvement = improvement / K
        
        loss = final_loss - lam * improvement
        
        info = {
            "loss": loss.item(),
            "final_loss": final_loss.item(),
            "improvement": improvement.item(),
            "J0_mean": J0.mean().item(),     # sanity check initial cost
            "JK_mean": costs[-1].mean().item(),  # final cost after K iters
        }

        return loss, info
    
    