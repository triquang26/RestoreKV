"""Read matching: the compressed cache should return what the full cache returns to the same queries.

For a teacher query q (post-RoPE, from the QA forward over the full cache) at one layer and head,
    a_T = Attn(q; context C  + QA prefix H)            (full cache)
    a_S = Attn(q; kept K + restore slots R_theta + H)  (compressed cache)
and the loss is ||a_S - sg(a_T)||^2 / (sg(||a_T||^2) + eps), averaged over sampled layers, heads and queries.
q, C, K and H are detached, so the gradient only reaches the restore pass that produced R_theta. Both reads
share q, H and positions; only the context part of the cache differs. The slot K/V stay free (no
constraint to be token averages): any representation that reads back correctly is optimal.
"""

import torch
import torch.nn.functional as F


def _read(q, keys, values, allowed, groups):
    """q (Hq, m, d); keys/values (Hkv, S, d); allowed (Hkv or 1, m, S) bool -> (Hq, m, d) in fp32."""
    keys, values = keys.repeat_interleave(groups, 0).float(), values.repeat_interleave(groups, 0).float()
    mask = allowed.repeat_interleave(groups, 0) if allowed.shape[0] > 1 else allowed
    return F.scaled_dot_product_attention(q.float()[None], keys[None], values[None], attn_mask=mask[None])[0]


def read_matching_loss(captured, kept, slots, rows, cols, context_length, eps: float = 1e-6) -> torch.Tensor:
    """captured[l] = (q (B, Hq, L, d), k (B, Hkv, T+L, d), v): teacher QA pass over the full cache;
    kept (num_layers, Hkv, T) bool; slots[l] = (sk, sv) (1, Hkv, n, d) with grad; (rows, cols) the query
    positions (row of the QA batch, position in that row)."""
    T, losses = context_length, []
    for layer, (q, k, v) in captured.items():
        groups = q.shape[1] // k.shape[1]
        sk, sv = slots[layer][0][0], slots[layer][1][0]
        for b in rows.unique().tolist():
            c = cols[rows == b]
            qb, L = q[b][:, c], k.shape[2] - T  # (Hq, m, d)
            causal = torch.arange(L, device=q.device)[None, :] <= c[:, None]  # (m, L): QA prefix up to the query
            full_ctx = torch.ones(len(c), T, dtype=torch.bool, device=q.device)
            a_t = _read(qb, k[b], v[b], torch.cat([full_ctx, causal], 1)[None], groups)
            kept_ctx = kept[layer][:, None, :].expand(-1, len(c), -1)  # (Hkv, m, T)
            allowed = torch.cat([kept_ctx, torch.ones(*kept_ctx.shape[:2], sk.shape[1], dtype=torch.bool, device=q.device),
                                 causal[None].expand(kept_ctx.shape[0], -1, -1)], -1)
            keys = torch.cat([k[b][:, :T], sk, k[b][:, T:]], 1)
            values = torch.cat([v[b][:, :T], sv, v[b][:, T:]], 1)
            a_s = _read(qb, keys, values, allowed, groups)
            err = (a_s - a_t).pow(2).sum(-1) / (a_t.pow(2).sum(-1) + eps)  # (Hq, m)
            losses.append(err.reshape(-1))
    return torch.cat(losses).mean()
