"""Hard attention masks for Partitioned Restore with Global Fusion (PRGF).

HF attention masks are shared by all layers and heads, but PRGF needs a different mask per
(layer, KV head) because the evictor keeps a different set of positions in each. We therefore
route SDPA through a small wrapper: inside ``attention_bias(provider)`` every attention layer asks
``provider(layer_idx, q_len, k_len)`` for a boolean mask of shape (1, num_heads, q_len, k_len)
(True = may attend) and calls the stock transformers SDPA kernel with it. Outside the context
manager the wrapper is a no-op.

Two masks are needed:

* restore pass (queries = n restore tokens [R_1..R_{n-1}, G], keys = context + restore tokens)
    - local R_j : kept context  +  evicted context in region P_j  +  itself
    - global G  : whole context  +  R_1..R_{n-1}  +  itself
  (``mode="causal"`` reproduces the original RestoreKV access pattern: full context, causal slots.)
    - PRGF v2 (``exchange_from_layer``): in the last layers R_j additionally reads G
      ("local encoding followed by global exchange").
* student pass (queries = question/answer tokens, keys = context + restore tokens + QA tokens)
    - kept context  +  all restore slots  +  causal over QA tokens
  i.e. exactly what the model sees after eviction, used for self-distillation.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Literal

import kvpress  # noqa: F401  (patches ALL_ATTENTION_FUNCTIONS first; we wrap on top of it)
import torch
from torch.utils.checkpoint import checkpoint
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

MaskMode = Literal["prgf", "causal"]
Partition = Literal["position", "evicted"]
MaskProvider = Callable[[int, int, int], torch.Tensor]

_PROVIDER: ContextVar[MaskProvider | None] = ContextVar("prgf_mask_provider", default=None)


@contextmanager
def attention_bias(provider: MaskProvider):
    """Use ``provider`` for every SDPA call made inside the block."""
    token = _PROVIDER.set(provider)
    try:
        yield
    finally:
        _PROVIDER.reset(token)


def _install():
    previous = ALL_ATTENTION_FUNCTIONS["sdpa"]
    if getattr(previous, "_prgf", False):
        return

    def sdpa_with_bias(module, query, key, value, attention_mask, *args, **kwargs):
        provider = _PROVIDER.get()
        if provider is None:
            slot_bias = getattr(module, "prgf_slot_bias", None)
            if slot_bias is not None and query.shape[-2] == key.shape[-2]:
                module.prgf_slot_bias = slot_bias = None  # a new prefill: the previous context's memory is gone
            if slot_bias is not None:  # transport-written slots enter attention with + log(mass)
                start, log_mass = slot_bias
                groups = query.shape[1] // log_mass.shape[0]
                attention_mask = slot_bias_mask(log_mass, start, query.shape[-2], key.shape[-2], groups, query.dtype)
            return previous(module, query, key, value, attention_mask, *args, **kwargs)

        def attend(q, k, v):
            mask = provider(module.layer_idx, q.shape[-2], k.shape[-2])
            if mask.is_floating_point():
                mask = mask.to(q.dtype)
            # Stock kernel (not the kvpress fake-key patch): keys stay untouched during these passes.
            return sdpa_attention_forward(module, q, k, v, mask, *args, **kwargs)[0]

        if torch.is_grad_enabled() and (query.requires_grad or key.requires_grad):
            # The dense mask is (num_heads, q_len, k_len); recompute it in backward instead of storing it.
            return checkpoint(attend, query, key, value, use_reentrant=False), None
        return attend(query, key, value), None

    sdpa_with_bias._prgf = True
    ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa_with_bias


_install()


def slot_bias_mask(log_mass: torch.Tensor, start: int, q_len: int, k_len: int, num_groups: int, dtype) -> torch.Tensor:
    """Additive mask (1, H_q, q_len, k_len): log(mass) on the n slot columns [start, start+n), causal on the
    last q_len columns (the new query tokens), 0 elsewhere."""
    device, n = log_mass.device, log_mass.shape[-1]
    mask = torch.zeros(log_mass.shape[0], q_len, k_len, dtype=torch.float32, device=device)
    mask[:, :, start : start + n] = log_mass.float()[:, None, :]
    if q_len > 1:
        future = torch.ones(q_len, q_len, dtype=torch.bool, device=device).triu(1)
        mask[:, :, k_len - q_len :] = mask[:, :, k_len - q_len :].masked_fill(future, float("-inf"))
    return mask.repeat_interleave(num_groups, dim=0)[None].to(dtype)


def region_ids(context_length: int, num_regions: int, device=None) -> torch.Tensor:
    """P_j = {t : floor(num_regions * t / T) = j}: contiguous, near-equal regions by token position."""
    return torch.arange(context_length, device=device) * num_regions // context_length


def evicted_region_ids(kept: torch.Tensor, num_regions: int) -> torch.Tensor:
    """Contiguous regions holding equal shares of the evicted KV pairs (counted over all leading dims, e.g.
    layers and heads): with e_t evicted pairs at position t and F(t) = sum_{u<t} e_u, P_j = {t : floor(R F(t) /
    F(T)) = j}. One partition per context, shared by every layer and head; depends only on the selection."""
    context_length = kept.shape[-1]
    evicted = (~kept).reshape(-1, context_length).sum(0).double()
    total = evicted.sum()
    if total == 0:
        return region_ids(context_length, num_regions, kept.device)
    before = evicted.cumsum(0) - evicted
    return (before * num_regions / total).floor().long().clamp_(max=num_regions - 1)


def partition_ids(kept: torch.Tensor, num_regions: int, partition: Partition = "position") -> torch.Tensor:
    if partition == "position":
        return region_ids(kept.shape[-1], num_regions, kept.device)
    if partition == "evicted":
        return evicted_region_ids(kept, num_regions)
    raise ValueError(f"Unknown partition {partition!r}")


def restore_allowed(
    kept: torch.Tensor, num_restore: int, mode: MaskMode = "prgf", slots_per_region: int = 1, num_global: int = 1,
    partition: Partition = "position",
) -> torch.Tensor:
    """Allowed-attention pattern of the restore tokens.

    kept: (..., num_kv_heads, T) bool, positions surviving eviction (any leading dims, e.g. layers).
    Slot layout (prgf): [region 0: k slots, region 1: k slots, ..., globals: g slots]. A local slot reads the
    kept cache, the evicted positions of its own region and the earlier slots of its own region (a causal
    chain, k = 1: only itself); a global slot reads the full context and every earlier slot.
    Returns (..., num_kv_heads, n, T + n) bool.
    """
    *lead, context_length = kept.shape
    n, device = num_restore, kept.device
    slots = torch.ones(n, n, dtype=torch.bool, device=device).tril()
    ctx = torch.ones(*lead, n, context_length, dtype=torch.bool, device=device)
    if mode == "prgf":
        n_local = n - num_global
        assert num_global >= 1 and n_local >= slots_per_region and n_local % slots_per_region == 0, (n, slots_per_region, num_global)
        slot_region = torch.arange(n_local, device=device) // slots_per_region
        regions = partition_ids(kept, n_local // slots_per_region, partition)
        own_region = regions[None, :] == slot_region[:, None]  # (n_local, T)
        ctx[..., :n_local, :] = kept[..., None, :] | own_region
        slots[:n_local, :n_local] &= slot_region[:, None] == slot_region[None, :]  # chains stay inside a region
    elif mode != "causal":
        raise ValueError(f"Unknown mask mode {mode!r}")
    return torch.cat([ctx, slots.expand(*lead, n, n)], dim=-1)


def student_allowed(kept: torch.Tensor, num_restore: int, q_len: int) -> torch.Tensor:
    """QA tokens after compression: kept context + all restore slots + causal QA. (num_kv_heads, q_len, T+n+q_len)."""
    num_kv_heads, _ = kept.shape
    device = kept.device
    ctx = kept[:, None, :].expand(-1, q_len, -1)
    slots = torch.ones(num_kv_heads, q_len, num_restore, dtype=torch.bool, device=device)
    qa = torch.ones(q_len, q_len, dtype=torch.bool, device=device).tril()[None].expand(num_kv_heads, -1, -1)
    return torch.cat([ctx, slots, qa], dim=-1)


def _to_query_heads(allowed: torch.Tensor, num_groups: int) -> torch.Tensor:
    # KV head h serves query heads h*G .. h*G+G-1 (same order as transformers' repeat_kv)
    return allowed.repeat_interleave(num_groups, dim=0)[None]


def exchange_layer(num_layers: int, exchange_from: float | None) -> int | None:
    """First layer of the local-global exchange stage: floor(exchange_from * L) (None = PRGF v1)."""
    return None if exchange_from is None else int(exchange_from * num_layers)


def restore_pass_provider(
    kept: torch.Tensor, num_restore: int, num_groups: int, mode: MaskMode, exchange_from_layer: int | None = None,
    slots_per_region: int = 1, num_global: int = 1, partition: Partition = "position",
) -> MaskProvider:
    """kept: (num_layers, num_kv_heads, T) bool. Masks of all layers are built at once (a few kernels).

    exchange_from_layer (PRGF v2): from this layer on, local slots also read the global slot G. G sits
    after the locals, so this edge is anti-causal; it is legal because the whole restore block is built
    from the context only (before any question), and Q/K/V of a layer come from that layer's inputs.
    """
    context_length = kept.shape[-1]
    allowed = restore_allowed(kept, num_restore, mode, slots_per_region, num_global, partition)  # (L, H_kv, n, T + n)
    if exchange_from_layer is not None:
        assert mode == "prgf", "local-global exchange extends the PRGF mask"
        n_local = num_restore - num_global
        allowed[exchange_from_layer:, :, :n_local, context_length + n_local :] = True  # locals -> globals

    def provider(layer_idx: int, q_len: int, k_len: int) -> torch.Tensor:
        assert q_len == num_restore and k_len == context_length + num_restore, (q_len, k_len)
        return _to_query_heads(allowed[layer_idx], num_groups)

    return provider


def student_pass_provider(
    kept: torch.Tensor, num_restore: int, num_groups: int, log_mass: torch.Tensor | None = None
) -> MaskProvider:
    """log_mass (num_layers, num_kv_heads, n): transport-written slots get + log(mass) on their logits."""
    context_length = kept.shape[-1]

    def provider(layer_idx: int, q_len: int, k_len: int) -> torch.Tensor:
        assert k_len == context_length + num_restore + q_len, (q_len, k_len)
        allowed = student_allowed(kept[layer_idx], num_restore, q_len)
        if log_mass is None:
            return _to_query_heads(allowed, num_groups)
        bias = torch.zeros(allowed.shape, dtype=torch.float32, device=allowed.device)
        bias[..., context_length : context_length + num_restore] = log_mass[layer_idx].float()[:, None, :]
        bias = bias.masked_fill(~allowed, float("-inf"))
        return _to_query_heads(bias, num_groups)

    return provider
