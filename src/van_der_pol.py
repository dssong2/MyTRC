import torch

class VanDerPol:
    def __init__(self, mu: float = 1.0, dt: float = 0.05):
        self.mu = mu
        self.dt = dt

    def f(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # x is of shape (batch_size, 2) where x[:, 0] is x1 and x[:, 1] is x2
        # u is of shape (batch_size, 1)
        x1, x2 = x[:, 0:1], x[:, 1:2]
        dx1 = x2
        dx2 = self.mu * (1 - x1**2) * x2 - x1 + u
        return torch.cat([dx1, dx2], dim=-1)
    
    def simulate(self, x0: torch.Tensor, u_seq: torch.Tensor) -> torch.Tensor:
        x = x0
        T = u_seq.shape[1]
        
        for t in range(T):
            u = u_seq[:, t, :]
            
            k1 = self.f(x, u)
            k2 = self.f(x + 0.5 * self.dt * k1, u)
            k3 = self.f(x + 0.5 * self.dt * k2, u)
            k4 = self.f(x + self.dt * k3, u)
            
            x = x + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
        return x
    
    def traj(self, x0: torch.Tensor, u_seq: torch.Tensor) -> torch.Tensor:
        x = x0
        T = u_seq.shape[1]
        traj = [x]
        
        for t in range(T):
            u = u_seq[:, t, :]
            
            k1 = self.f(x, u)
            k2 = self.f(x + 0.5 * self.dt * k1, u)
            k3 = self.f(x + 0.5 * self.dt * k2, u)
            k4 = self.f(x + self.dt * k3, u)
            
            x = x + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            traj.append(x)
            
        return torch.stack(traj, dim=1)