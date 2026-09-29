"""Transport write: conservation, scope, monotone objective, decode bias semantics, end-to-end press/trainer."""

import torch

from prgf.masking import region_ids, slot_bias_mask
from prgf.transport import TransportConfig, slot_scope, transport_write


def _setup(H=2, T=70, d=8, n_regions=7, k=2, g=2, seed=0):
    torch.manual_seed(seed)
    n = n_regions * k + g
    keys, values = torch.randn(H, T, d), torch.randn(H, T, d)
    mu0, nu0 = torch.randn(H, n, d), torch.randn(H, n, d)
    evicted = torch.rand(H, T) < 0.8
    A = torch.randn(d, d)
    G = (A @ A.T / d).expand(H, d, d)
    return keys, values, mu0, nu0, evicted, slot_scope(n_regions, k, g, T), G


def test_conservation_and_scope():
    keys, values, mu0, nu0, evicted, scope, G = _setup()
    mu, nu, log_mass, P, _ = transport_write(keys, values, mu0, nu0, evicted, scope, G, TransportConfig(iters=2), return_objective=True)
    mass = log_mass.exp()
    for h in range(keys.shape[0]):
        E = evicted[h]
        torch.testing.assert_close(mass[h].sum(), E.sum().float())  # every evicted token = one unit of mass
        torch.testing.assert_close((mass[h, :, None] * mu[h]).sum(0), keys[h, E].sum(0), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close((mass[h, :, None] * nu[h]).sum(0), values[h, E].sum(0), rtol=1e-4, atol=1e-4)
        assert (P[h][:, ~E] == 0).all()  # kept tokens carry no mass
        assert (P[h][~scope] == 0).all()  # local slots only take tokens of their own region
        torch.testing.assert_close(P[h][:, E].sum(0), torch.ones(int(E.sum())))


def test_objective_does_not_increase():
    keys, values, mu0, nu0, evicted, scope, G = _setup(seed=1)
    *_, objectives = transport_write(keys, values, mu0, nu0, evicted, scope, G, TransportConfig(iters=5), return_objective=True)
    assert all(b <= a + 1e-4 * abs(a) for a, b in zip(objectives, objectives[1:])), objectives


def test_empty_slot_is_masked():
    keys, values, mu0, nu0, evicted, scope, G = _setup(seed=2)
    evicted[:, region_ids(keys.shape[1], 7) == 3] = False  # nothing evicted in region 3 -> its local slots are empty
    mu, nu, log_mass = transport_write(keys, values, mu0, nu0, evicted, scope, G, TransportConfig())
    assert torch.isinf(log_mass[:, 6:8]).all() and (log_mass[:, 6:8] < 0).all()
    assert torch.isfinite(mu).all() and torch.isfinite(nu).all()


def test_log_mass_bias_equals_duplicated_keys():
    torch.manual_seed(0)
    H, d, T, n, q_len = 2, 8, 5, 2, 3
    q = torch.randn(1, H, q_len, d)
    ctx_k, ctx_v = torch.randn(1, H, T, d), torch.randn(1, H, T, d)
    slot_k, slot_v = torch.randn(1, H, n, d), torch.randn(1, H, n, d)
    qa_k, qa_v = torch.randn(1, H, q_len, d), torch.randn(1, H, q_len, d)
    mass = torch.tensor([[3.0, 2.0], [1.0, 4.0]])
    k = torch.cat([ctx_k, slot_k, qa_k], 2)
    v = torch.cat([ctx_v, slot_v, qa_v], 2)
    mask = slot_bias_mask(mass.log(), T, q_len, T + n + q_len, 1, torch.float32)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    for h in range(H):  # reference: slot j repeated mass[h, j] times, causal over the query tokens
        reps = [slot_k[:, h, j : j + 1].repeat(1, int(mass[h, j]), 1) for j in range(n)]
        repv = [slot_v[:, h, j : j + 1].repeat(1, int(mass[h, j]), 1) for j in range(n)]
        kk, vv = torch.cat([ctx_k[:, h], *reps, qa_k[:, h]], 1), torch.cat([ctx_v[:, h], *repv, qa_v[:, h]], 1)
        causal = torch.ones(q_len, kk.shape[1], dtype=torch.bool)
        causal[:, -q_len:] = torch.ones(q_len, q_len, dtype=torch.bool).tril()
        ref = torch.nn.functional.scaled_dot_product_attention(q[:, h], kk, vv, attn_mask=causal)
        torch.testing.assert_close(out[:, h], ref, rtol=1e-5, atol=1e-5)
