import time
import numpy as np
from scipy.optimize import minimize
import torch

class SQP:
    def __init__(
        self, T: int, dt: float, mu: float, u_min: float, u_max: float,
        Q: np.ndarray , R: float, Qf: np.ndarray
    ):
        self.T = T
        self.dt = dt
        self.mu = mu
        self.u_min = u_min
        self.u_max = u_max
        self.Q = Q
        self.R = R
        self.Qf = Qf
        
    def f_numpy(self, x, u):
        x1, x2 = x[0], x[1]
        dx1 = x2
        dx2 = self.mu * (1 - x1**2) * x2 - x1 + u
        return np.array([dx1, dx2])

    def rk4_step(self, x, u):
        k1 = self.f_numpy(x, u)
        k2 = self.f_numpy(x + 0.5*self.dt*k1, u)
        k3 = self.f_numpy(x + 0.5*self.dt*k2, u)
        k4 = self.f_numpy(x +     self.dt*k3, u)
        return x + (self.dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def simulate_numpy(self, x0, u_seq):
        traj = [x0]
        x = x0.copy()
        for t in range(self.T):
            x = self.rk4_step(x, u_seq[t])
            traj.append(x)
        return np.array(traj)
    
    # Check this
    def cost(self, u_seq, x0):
        traj = self.simulate_numpy(x0, u_seq)
        cost = 0.0
        for t in range(self.T):
            x_t : np.ndarray = traj[t]
            u_t : float = u_seq[t]
            cost += x_t.T @ self.Q @ x_t + self.R * (u_t**2)
        x_T : np.ndarray = traj[-1]
        cost += x_T.T @ self.Qf @ x_T
        return cost
    
    # def solve_single(self, x0):
    #     u_init = np.zeros(self.T)  # Initial guess for control sequence
    #     bounds = [(self.u_min, self.u_max) for _ in range(self.T)]
        
    #     result = minimize(
    #         fun=self.cost,
    #         x0=u_init,
    #         args=(x0,),
    #         method='SLSQP',
    #         bounds=bounds,
    #         options={
    #             'ftol': 1e-8,
    #             'maxiter': 1000,
    #             'disp': False,  
    #         }
    #     )
        
    #     u_star = result.x.reshape(self.T, 1)
    #     cost = result.fun
        
    #     return u_star, cost
    
    def solve_single(self, x0):
        """
        Solve with multiple initializations and return the best result.
        Addresses poor local minima for initial conditions near the limit cycle.
        """
        best_u    = None
        best_cost = np.inf

        # Generate several different warm starts
        initializations = self._get_initializations(x0)

        for u_init in initializations:
            result = minimize(
                fun=self.cost,
                x0=u_init,
                args=(x0,),
                method='SLSQP',
                bounds=[(self.u_min, self.u_max)] * self.T,
                options={
                    'ftol': 1e-9,       # tighter tolerance than before
                    'maxiter': 2000,    # more iterations per solve
                    'disp': False,
                }
            )

            if result.fun < best_cost:
                best_cost = result.fun
                best_u    = result.x

        return best_u.reshape(self.T, 1), best_cost


    def _get_initializations(self, x0):
        """
        Generate multiple starting points for multi-start SQP.
        Each captures a different strategy for driving the system to origin.
        """
        inits = []

        # 1. Zero initialization — neutral starting point
        inits.append(np.zeros(self.T))

        # 2. Negative of initial position scaled
        # If x0[0] > 0 apply negative control, vice versa
        # Simple proportional strategy
        prop = np.ones(self.T) * (-x0[0] * 2.0)
        inits.append(np.clip(prop, self.u_min, self.u_max))

        # 3. Linear decay from max control to zero
        # Aggressive early control then coast
        sign    = -np.sign(x0[0]) if x0[0] != 0 else 1.0
        decay   = np.linspace(self.u_max, 0.0, self.T) * sign
        inits.append(np.clip(decay, self.u_min, self.u_max))

        # 4. Bang-bang style — full control for first half, zero second half
        # Mimics the fuel-optimal structure
        bang         = np.zeros(self.T)
        bang[:self.T//2] = self.u_max * sign
        inits.append(bang)

        # 5. Small random perturbation around zero
        # Helps escape symmetric saddle points
        rng = np.random.default_rng(int(abs(x0[0] * 1000)))
        inits.append(rng.uniform(-0.5, 0.5, self.T))

        return inits
    
    def generate_dataset(self, N: int, seed: int = 42):
        rng = np.random.default_rng(seed)

        x0_all       = rng.uniform(low=-2.0, high=2.0, size=(N, 2))
        x_target_all = np.zeros((N, 2))
        u_star_all   = np.zeros((N, self.T, 1))
        costs        = np.zeros(N)
        failed       = []

        t_start = time.time()

        for i in range(N):
            if i % 500 == 0 and i > 0:
                elapsed   = time.time() - t_start
                per_iter  = elapsed / i
                remaining = per_iter * (N - i)
                mean_so_far = costs[:i].mean()
                print(
                    f"  Solving {i}/{N} — "
                    f"{remaining/60:.1f} min remaining — "
                    f"mean cost so far: {mean_so_far:.2f}"
                )
            elif i == 0:
                print(f"  Solving {i}/{N}...")

            u_star, cost = self.solve_single(x0_all[i])
            u_star_all[i] = u_star
            costs[i]      = cost

            if cost > 1e6:
                failed.append(i)

        if failed:
            print(f"Warning: {len(failed)} solves may have failed: {failed[:10]}")

        print(f"Cost statistics:")
        print(f"  Mean: {costs.mean():.2f}")
        print(f"  Std:  {costs.std():.2f}")
        print(f"  Min:  {costs.min():.2f}")
        print(f"  Max:  {costs.max():.2f}")

        return {
            'x0':       x0_all,
            'x_target': x_target_all,
            'costs':    costs,
            'u_star':   u_star_all,
        }
        
    def save_dataset(self, dataset, path):
        """Save as a PyTorch .pt file for direct loading in training."""
        torch_dataset = {
            'x0': torch.tensor(dataset['x0'], dtype=torch.float32),
            'x_target': torch.tensor(dataset['x_target'], dtype=torch.float32),
            'u_star': torch.tensor(dataset['u_star'], dtype=torch.float32),
            'costs': torch.tensor(dataset['costs'], dtype=torch.float32),
        }
        torch.save(torch_dataset, path)
        print(f"Saved {len(dataset['x0'])} samples to {path}")


    def load_dataset(self, path):
        """Load dataset saved by save_dataset."""
        return torch.load(path, weights_only=False)
    
    def validate_single_solve(self):
        # Test several initial conditions including ones near the limit cycle
        test_points = [
            np.array([1.5,  1.0]),    # original test point
            np.array([2.0,  2.0]),    # corner of the space — hardest case
            np.array([-1.5, 1.5]),    # near limit cycle
            np.array([0.1,  0.1]),    # near origin — should be easy
        ]

        for x0 in test_points:
            u_star, cost = self.solve_single(x0)
            traj = self.simulate_numpy(x0, u_star.flatten())
            x_T  = traj[-1]

            print(f"x0={x0}  →  x_T={x_T.round(4)}  cost={cost:.4f}  "
                f"bounds_ok={bool((u_star >= self.u_min).all() and (u_star <= self.u_max).all())}")


def main():
    sqp = SQP(
        T=100,
        dt=0.05,
        mu=1.0,
        u_min=-2.0,
        u_max=2.0,
        Q=np.diag([10.0, 5.0]),
        R=0.5,
        Qf=20.0 * np.diag([10.0, 5.0]),
    )

    # Validate one solve before generating the full set
    print("Validating single solve...")
    sqp.validate_single_solve()

    # # Generate training set
    # print("\nGenerating training set (10,000 samples)...")
    # train_data = sqp.generate_dataset(N=10_000, seed=42)
    # sqp.save_dataset(dataset=train_data, path="van_der_pol_train.pt")

    # # Generate test set
    # print("\nGenerating test set (1,000 samples)...")
    # test_data = sqp.generate_dataset(N=1_000, seed=99)
    # sqp.save_dataset(dataset=test_data, path="van_der_pol_test.pt")
        
        
if __name__ == "__main__":
    main()