"""Transport write: turn evicted KV into slot memory by a conserving, scope-restricted assignment.

For one layer and every KV head h (vectorised), with evicted set E_h and slots j = 1..n:

    min_P  sum_{j,t} P_jt C_jt + tau * sum_{j,t} P_jt log P_jt
    s.t.   P_jt >= 0,  sum_j P_jt = 1 for t in E_h,  P_jt = 0 outside slot j's scope

    C_jt = (k_t - mu_j)^T G (k_t - mu_j) / s_k + lambda_v * ||v_t - nu_j||^2 / s_v

G = E[q q^T / d] over reference queries, so the key term is the expected squared attention-logit error of
replacing k_t by mu_j; s_k, s_v are per-(layer, head) scales of the initial costs (held fixed across rounds)
so that tau and lambda_v are dimensionless. With C fixed, the exact minimiser is a softmax over each token's
admissible slots; with P fixed, the exact minimiser of the representatives is the P-weighted mean. Each round
therefore does not increase the objective. The resulting memory conserves mass and first moments:

    sum_j m_j = |E|,   sum_j m_j mu_j = sum_{t in E} k_t,   sum_j m_j nu_j = sum_{t in E} v_t,

and a slot enters attention with logit q^T mu_j / sqrt(d) + log m_j (log 0 = -inf masks empty slots).
Keys are post-RoPE cache keys; nothing is re-rotated.
"""

from dataclasses import dataclass

import torch


@dataclass
class TransportConfig:
    iters: int = 2
    tau: float = 0.1
    lambda_v: float = 1.0


def slot_scope(num_regions: int, slots_per_region: int, num_global: int, context_length: int, device=None) -> torch.Tensor:
    """(n, T) bool: token t may be assigned to slot j (local slots: own region only; globals: everywhere)."""
    from prgf.masking import region_ids

    regions = region_ids(context_length, num_regions, device)
    slot_region = torch.arange(num_regions * slots_per_region, device=device) // slots_per_region
    local = regions[None, :] == slot_region[:, None]
    glob = torch.ones(num_global, context_length, dtype=torch.bool, device=device)
    return torch.cat([local, glob])


def _costs(keys, values, mu, nu, G):
    """Key term (k-mu)^T G (k-mu) and value term ||v-nu||^2, both (H, n, T)."""
    Gk = keys @ G  # (H, T, d)
    kGk = (keys * Gk).sum(-1)  # (H, T)
    muG = mu @ G  # (H, n, d)
    ck = kGk[:, None, :] - 2 * muG @ keys.transpose(1, 2) + (muG * mu).sum(-1)[:, :, None]
    cv = (values * values).sum(-1)[:, None, :] - 2 * nu @ values.transpose(1, 2) + (nu * nu).sum(-1)[:, :, None]
    return ck.clamp_min(0), cv.clamp_min(0)


def transport_write(
    keys: torch.Tensor, values: torch.Tensor, mu0: torch.Tensor, nu0: torch.Tensor, evicted: torch.Tensor,
    scope: torch.Tensor, G: torch.Tensor, cfg: TransportConfig, return_objective: bool = False,
):
    """keys/values (H, T, d) post-RoPE context KV; mu0/nu0 (H, n, d) initial slot KV (from the PRGF restore pass);
    evicted (H, T) bool; scope (n, T) bool; G (H, d, d). Returns mu, nu (H, n, d) and log_mass (H, n)."""
    in_dtype = mu0.dtype
    keys, values, mu, nu, G = (x.float() for x in (keys, values, mu0, nu0, G))
    admissible = evicted[:, None, :] & scope[None]  # (H, n, T)
    ck, cv = _costs(keys, values, mu, nu, G)
    count = admissible.sum((1, 2)).clamp_min(1)
    s_k = ((ck * admissible).sum((1, 2)) / count).detach().clamp_min(1e-6)[:, None, None]
    s_v = ((cv * admissible).sum((1, 2)) / count).detach().clamp_min(1e-6)[:, None, None]
    objectives = []
    for _ in range(cfg.iters):
        cost = ck / s_k + cfg.lambda_v * cv / s_v
        logits = (-cost / cfg.tau).masked_fill(~admissible, float("-inf"))
        P = torch.softmax(logits, dim=1).nan_to_num(0.0) * evicted[:, None, :]  # kept tokens carry no mass
        mass = P.sum(-1)  # (H, n)
        safe = mass.clamp_min(1e-12)[..., None]
        new_mu, new_nu = (P @ keys) / safe, (P @ values) / safe
        empty = (mass <= 1e-12)[..., None]
        mu, nu = torch.where(empty, mu, new_mu), torch.where(empty, nu, new_nu)
        ck, cv = _costs(keys, values, mu, nu, G)
        if return_objective:
            ent = (P * P.clamp_min(1e-30).log()).sum()
            objectives.append(((P * (ck / s_k + cfg.lambda_v * cv / s_v)).sum() + cfg.tau * ent).item())
    log_mass = mass.log()  # -inf for empty slots
    out = (mu.to(in_dtype), nu.to(in_dtype), log_mass)
    return (*out, P, objectives) if return_objective else out
