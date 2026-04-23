import torch
import torch.utils.data as data
from src.trc import TRC
from src.train import VanDerPolDataset, CONFIG


def evaluate(checkpoint_path, test_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load checkpoint ----
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    config = checkpoint["config"]

    # ---- Reconstruct model ----
    model = TRC(
        d_x      = config["d_x"],
        d_u      = config["d_u"],
        T        = config["T"],
        K        = config["K"],
        n        = config["n"],
        d_z      = config["d_z"],
        d_h      = config["d_h"],
        L        = config["L"],
        num_heads= config["num_heads"],
        u_min    = config["u_min"],
        u_max    = config["u_max"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # ---- Cost matrices ----
    Q  = config["Q"].to(device)
    Qf = config["Qf"].to(device)
    R  = config["R"]

    # ---- Test dataset ----
    test_dataset = VanDerPolDataset(test_path)
    test_loader  = data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # ---- Load SQP reference costs from dataset ----
    raw_dataset = torch.load(test_path, weights_only=False)
    sqp_costs   = raw_dataset["costs"]   # (N,) — optimal costs from SQP

    # ---- Run TRC on test set ----
    trc_costs_all = []
    sqp_costs_all = []

    # Also track per-iteration costs to validate refinement
    iter_costs_all = [[] for _ in range(config["K"] + 1)]

    with torch.no_grad():
        for batch_idx, (x0, x_target, u_star) in enumerate(test_loader):
            x0       = x0.to(device)
            x_target = x_target.to(device)

            # Get all intermediate control sequences
            all_u = model(
                x0, x_target,
                dynamics_fn=model.van_der_pol.simulate,
                return_all_iters=True,
            )

            # Compute cost for each iteration
            for k, u_k in enumerate(all_u):
                traj = model.van_der_pol.traj(x0, u_k)
                J_k  = model.cost(traj, u_k, Q, R, Qf)   # (B,)
                iter_costs_all[k].append(J_k.cpu())

            # Final TRC cost is the last iteration
            trc_costs_all.append(iter_costs_all[-1][-1])

        # Concatenate across batches
        trc_costs  = torch.cat(trc_costs_all)              # (N,)
        iter_costs = [
            torch.cat(iter_costs_all[k]) for k in range(config["K"] + 1)
        ]                                                   # K+1 tensors of (N,)

    # ---- Compare against SQP ----
    ratio = trc_costs / sqp_costs   # (N,) — per-sample ratio

    print("=" * 50)
    print("TRC vs SQP Cost Comparison")
    print("=" * 50)
    print(f"{'':20s}  {'Mean':>10}  {'Std':>10}  {'Min':>10}  {'Max':>10}")
    print("-" * 60)
    print(
        f"{'SQP (optimal)':20s}  "
        f"{sqp_costs.mean():>10.2f}  "
        f"{sqp_costs.std():>10.2f}  "
        f"{sqp_costs.min():>10.2f}  "
        f"{sqp_costs.max():>10.2f}"
    )
    print(
        f"{'TRC (final iter)':20s}  "
        f"{trc_costs.mean():>10.2f}  "
        f"{trc_costs.std():>10.2f}  "
        f"{trc_costs.min():>10.2f}  "
        f"{trc_costs.max():>10.2f}"
    )
    print(
        f"{'TRC/SQP ratio':20s}  "
        f"{ratio.mean():>10.4f}  "
        f"{ratio.std():>10.4f}  "
        f"{ratio.min():>10.4f}  "
        f"{ratio.max():>10.4f}"
    )

    # ---- Per-iteration cost reduction ----
    # This validates Figure 4b from the paper
    print("\nCost reduction across refinement iterations:")
    print(f"{'Iteration':>12}  {'Mean Cost':>12}  {'% of Iter 0':>12}")
    print("-" * 40)
    J0_mean = iter_costs[0].mean().item()
    for k, costs_k in enumerate(iter_costs):
        mean_k   = costs_k.mean().item()
        pct_of_0 = mean_k / J0_mean * 100
        print(f"{k:>12}  {mean_k:>12.2f}  {pct_of_0:>11.1f}%")

    # ---- How close to optimal ----
    pct_above_optimal = ((trc_costs - sqp_costs) / sqp_costs * 100)
    print(f"\nTRC is on average {pct_above_optimal.mean():.2f}% above SQP optimal")
    print(f"Paper claims ~0% gap (exact match) on Van der Pol")


if __name__ == "__main__":
    evaluate(
        checkpoint_path="trc_checkpoint.pt",
        test_path="van_der_pol_test.pt",
    )