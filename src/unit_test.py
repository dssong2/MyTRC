import torch
import torch.nn as nn
import pytest
from src.trc import TRC

# -----------------------------------------------------------------------
# Shared fixtures — problem dimensions matching paper Section V-A
# -----------------------------------------------------------------------

B     = 4       # batch size
T     = 100     # control horizon
d_x   = 2       # Van der Pol state dimension
d_u   = 1       # Van der Pol control dimension
d_z   = 256     # latent dimension
d_h   = 512     # hidden dimension
K     = 3       # refinement iterations
n     = 4       # inner cycles
u_min = -2.0
u_max =  2.0

Q  = torch.tensor([[10., 0.], [0., 5.]])
Qf = 20.0 * Q
R  = 0.5


def make_model():
    return TRC(
        d_x=d_x, d_u=d_u, T=T, K=K, n=n,
        d_z=d_z, d_h=d_h, L=3, num_heads=8,
        u_min=u_min, u_max=u_max
    )


def make_batch():
    x0      = torch.rand(B, d_x) * 4 - 2   # uniform [-2, 2]^2
    x_target = torch.zeros(B, d_x)
    u_star  = torch.zeros(B, T, d_u)
    return x0, x_target, u_star


def dummy_dynamics(x0, u_seq):
    """Minimal dynamics stub — returns x0 unchanged. Just for shape testing."""
    return x0


# -----------------------------------------------------------------------
# 1. Construction tests
# -----------------------------------------------------------------------

def test_model_constructs():
    """Model should construct without errors."""
    model = make_model()
    assert model is not None


def test_parameter_count():
    """
    Paper Section III-A targets ~1.5M parameters for d_z=256, d_h=512, L=3.
    Test that count is in a reasonable range — not wildly over or under.
    """
    model = make_model()
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")
    assert 1_000_000 < total < 5_000_000, \
        f"Parameter count {total} outside expected range [1M, 5M]"


def test_all_submodules_registered():
    """
    All five neural network components plus H_init/L_init must be
    registered so optimizer sees them. Tests nn.Module registration.
    """
    model = make_model()
    param_names = [name for name, _ in model.named_parameters()]

    expected = [
        "state_encoder",
        "error_embed",
        "ctrl_embed",
        "reasoning",
        "initial_decoder",
        "residual_decoder",
        "H_init",
        "L_init",
        "H_proj",
        "L_proj",
    ]
    for name in expected:
        assert any(name in p for p in param_names), \
            f"'{name}' not found in registered parameters — " \
            f"check it's assigned via self.{name} and uses nn.Module or nn.Parameter"


# -----------------------------------------------------------------------
# 2. _init_latents tests
# -----------------------------------------------------------------------

def test_init_latents_shape():
    """
    _init_latents should return z_H and z_L both of shape (B, d_z).
    Bug to catch: z0.size[0] should be z0.shape[0]
    """
    model = make_model()
    z0 = torch.randn(B, d_z)
    zH, zL = model._init_latents(z0)

    assert zH.shape == (B, d_z), \
        f"z_H shape {zH.shape} — expected ({B}, {d_z})"
    assert zL.shape == (B, d_z), \
        f"z_L shape {zL.shape} — expected ({B}, {d_z})"


def test_init_latents_sample_specific():
    """
    Different z0 inputs should produce different latents.
    If H_proj/L_proj aren't applied, all samples would get identical latents.
    """
    model = make_model()
    z0_a = torch.randn(B, d_z)
    z0_b = torch.randn(B, d_z)

    zH_a, zL_a = model._init_latents(z0_a)
    zH_b, zL_b = model._init_latents(z0_b)

    assert not torch.allclose(zH_a, zH_b), \
        "z_H is identical for different z0 — H_proj may not be applied"
    assert not torch.allclose(zL_a, zL_b), \
        "z_L is identical for different z0 — L_proj may not be applied"


# -----------------------------------------------------------------------
# 3. Forward pass shape tests
# -----------------------------------------------------------------------

def test_forward_output_shape_final_only():
    """
    When return_all_iters=False, forward should return u of shape (B, T, d_u).
    Bug to catch: initial_decoder output dim set to d_u instead of T*d_u.
    """
    model = make_model()
    x0, x_target, _ = make_batch()

    u = model(x0, x_target, dummy_dynamics, return_all_iters=False)

    assert u.shape == (B, T, d_u), \
        f"Final u shape {u.shape} — expected ({B}, {T}, {d_u}). " \
        f"Check initial_decoder out_dim is T*d_u not d_u"


def test_forward_output_shape_all_iters():
    """
    When return_all_iters=True, forward should return list of K+1 tensors
    each of shape (B, T, d_u).
    """
    model = make_model()
    x0, x_target, _ = make_batch()

    all_u = model(x0, x_target, dummy_dynamics, return_all_iters=True)

    assert isinstance(all_u, list), \
        f"Expected list, got {type(all_u)}"
    assert len(all_u) == K + 1, \
        f"Expected {K+1} iterations (u^0..u^{K}), got {len(all_u)}"
    for k, u_k in enumerate(all_u):
        assert u_k.shape == (B, T, d_u), \
            f"u^({k}) shape {u_k.shape} — expected ({B}, {T}, {d_u})"


def test_forward_control_bounds_respected():
    """
    All control values in every iteration must stay within [u_min, u_max].
    Paper enforces this via clamp in Eq. (11).
    """
    model = make_model()
    x0, x_target, _ = make_batch()

    all_u = model(x0, x_target, dummy_dynamics, return_all_iters=True)

    for k, u_k in enumerate(all_u):
        assert u_k.min() >= u_min - 1e-6, \
            f"u^({k}) violates lower bound {u_min}: min={u_k.min():.4f}"
        assert u_k.max() <= u_max + 1e-6, \
            f"u^({k}) violates upper bound {u_max}: max={u_k.max():.4f}"


def test_ctrl_embed_called_correctly():
    """
    z_ctx = z0 + error_embed(e) + ctrl_embed(u.flatten(1))
    Bug to catch: ctrl_embed missing the call — self.ctrl_embed instead of
    self.ctrl_embed(u.flatten(1)) — which passes the module object not its output.
    This would cause a TypeError during forward.
    """
    model = make_model()
    x0, x_target, _ = make_batch()

    try:
        all_u = model(x0, x_target, dummy_dynamics, return_all_iters=True)
    except TypeError as e:
        raise AssertionError(
            f"Forward pass raised TypeError — likely ctrl_embed called without "
            f"input argument: self.ctrl_embed instead of self.ctrl_embed(u.flatten(1))\n"
            f"Original error: {e}"
        )


# -----------------------------------------------------------------------
# 4. Cost function tests
# -----------------------------------------------------------------------

def test_cost_shape():
    """cost() should return (B,) — one scalar cost per sample."""
    model = make_model()
    x0, _, _ = make_batch()
    u_seq = torch.zeros(B, T, d_u)
    traj  = model.van_der_pol.traj(x0, u_seq)

    c = model.cost(traj, u_seq, Q, R, Qf)

    assert c.shape == (B,), \
        f"Cost shape {c.shape} — expected ({B},)"


def test_cost_nonnegative():
    """
    Cost must be nonnegative since Q, Qf are PSD and R > 0.
    A negative cost indicates a sign error in the implementation.
    """
    model = make_model()
    x0, _, _ = make_batch()
    u_seq = torch.randn(B, T, d_u).clamp(u_min, u_max)
    traj  = model.van_der_pol.traj(x0, u_seq)

    c = model.cost(traj, u_seq, Q, R, Qf)

    assert (c >= 0).all(), \
        f"Cost has negative values: {c} — check Q, Qf are PSD and R > 0"


def test_cost_zero_at_origin_zero_control():
    """
    If x0 is already at the origin and control is zero,
    cost should be zero since x_t=0 for all t and u_t=0 for all t.
    """
    model = make_model()
    x0    = torch.zeros(B, d_x)
    u_seq = torch.zeros(B, T, d_u)
    traj  = model.van_der_pol.traj(x0, u_seq)

    c = model.cost(traj, u_seq, Q, R, Qf)

    assert torch.allclose(c, torch.zeros(B), atol=1e-4), \
        f"Cost at origin with zero control should be 0, got {c}"


def test_cost_gradient_flows():
    model = make_model()
    x0    = torch.rand(B, d_x) * 4 - 2
    u_seq = torch.randn(B, T, d_u).clamp(u_min, u_max)
    u_seq.requires_grad_(True)    # set on leaf tensor after clamp
    traj  = model.van_der_pol.traj(x0, u_seq)

    c = model.cost(traj, u_seq, Q, R, Qf)
    c.mean().backward()

    assert u_seq.grad is not None, \
        "No gradient reached u_seq — check van_der_pol.traj uses only PyTorch operations"
    assert u_seq.grad.norm() > 0, \
        "Gradient is zero — cost may not depend on u_seq"


# -----------------------------------------------------------------------
# 5. Loss function tests
# -----------------------------------------------------------------------

def test_loss_improvement_nonzero_for_improving_sequences():
    """
    Core test: improvement reward must be positive when each iteration
    produces genuinely better controls than the previous.
    Paper targets improvement metric ~0.32 at convergence (Figure 2).
    """
    model = make_model()
    x0, _, u_star = make_batch()

    # Manually construct improving sequence — scaling down toward zero
    all_u = [
        torch.ones(B, T, d_u) * scale
        for scale in [2.0, 1.0, 0.5, 0.1]
    ]

    loss, info = model.loss(all_u, u_star, x0, Q, R, Qf, lam=0.1)

    print(f"Improvement: {info['improvement']:.4f}")
    print(f"J0 mean:     {info['J0_mean']:.4f}")
    print(f"JK mean:     {info['JK_mean']:.4f}")

    assert info['improvement'] > 0, \
        f"Improvement {info['improvement']:.4f} should be positive " \
        f"for strictly improving sequences"


def test_loss_improvement_zero_for_identical_sequences():
    """
    If all iterations are identical, improvement must be zero.
    J_tilde(k-1) - J_tilde(k) = 0 for all k.
    """
    model = make_model()
    x0, _, u_star = make_batch()

    u_same = torch.ones(B, T, d_u) * 0.5
    all_u  = [u_same] * (K + 1)

    loss, info = model.loss(all_u, u_star, x0, Q, R, Qf, lam=0.1)

    assert abs(info['improvement']) < 1e-4, \
        f"Improvement should be ~0 for identical sequences, got {info['improvement']:.6f}"


def test_loss_output_shape_and_type():
    """
    loss() should return (tensor, dict).
    Tensor must be scalar for .backward() to work.
    """
    model = make_model()
    x0, _, u_star = make_batch()

    all_u = [torch.randn(B, T, d_u) for _ in range(K + 1)]
    loss, info = model.loss(all_u, u_star, x0, Q, R, Qf, lam=0.1)

    assert isinstance(loss, torch.Tensor), \
        f"loss should be a tensor, got {type(loss)}"
    assert loss.shape == torch.Size([]), \
        f"loss should be scalar, got shape {loss.shape}"
    assert isinstance(info, dict), \
        f"info should be dict, got {type(info)}"

    expected_keys = ["loss", "final_loss", "improvement", "J0_mean", "JK_mean"]
    for key in expected_keys:
        assert key in info, f"Missing key '{key}' in info dict"


def test_loss_gradient_flows_to_model_parameters():
    """
    The complete end-to-end gradient check.
    After loss.backward(), every model parameter should have a gradient.
    This is the final verification before connecting to the training loop.
    """
    model = make_model()
    x0, x_target, u_star = make_batch()

    all_u = model(x0, x_target, dummy_dynamics, return_all_iters=True)
    loss, info = model.loss(all_u, u_star, x0, Q, R, Qf, lam=0.1)
    loss.backward()

    missing_grads = []
    for name, param in model.named_parameters():
        if param.grad is None:
            missing_grads.append(name)

    if missing_grads:
        raise AssertionError(
            f"The following parameters received no gradient:\n"
            + "\n".join(f"  {n}" for n in missing_grads)
        )


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_model_constructs,
        test_parameter_count,
        test_all_submodules_registered,
        test_init_latents_shape,
        test_init_latents_sample_specific,
        test_forward_output_shape_final_only,
        test_forward_output_shape_all_iters,
        test_forward_control_bounds_respected,
        test_ctrl_embed_called_correctly,
        test_cost_shape,
        test_cost_nonnegative,
        test_cost_zero_at_origin_zero_control,
        test_cost_gradient_flows,
        test_loss_improvement_nonzero_for_improving_sequences,
        test_loss_improvement_zero_for_identical_sequences,
        test_loss_output_shape_and_type,
        test_loss_gradient_flows_to_model_parameters,
    ]

    passed, failed = 0, []
    for test in tests:
        try:
            test()
            print(f"  PASSED  {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAILED  {test.__name__}")
            print(f"          {e}")
            failed.append(test.__name__)

    print(f"\n{passed}/{len(tests)} passed")
    if failed:
        print("Failed tests:")
        for name in failed:
            print(f"  {name}")