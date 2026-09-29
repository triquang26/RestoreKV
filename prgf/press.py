"""KVPress press for Partitioned Restore with Global Fusion (PRGF)."""

import hashlib
import os
from dataclasses import dataclass, field

import torch
from huggingface_hub import hf_hub_download
from kvpress import KVzipPress, RestoreKVPress
from safetensors.torch import load_file
from transformers import PreTrainedModel

from prgf.masking import MaskMode, attention_bias, exchange_layer, restore_pass_provider
from prgf.transport import TransportConfig, slot_scope, transport_write

EMBEDDINGS_FILE = "restore_embeddings.safetensors"


def resolve_file(adapter: str, filename: str) -> str:
    """``adapter`` is either a local directory or a Hugging Face repo id."""
    if os.path.isdir(adapter):
        return os.path.join(adapter, filename)
    return hf_hub_download(adapter, filename)


def official_adapter(model: PreTrainedModel, plus: bool = False) -> str:
    return f"higokri/RestoreKV-{model.config.name_or_path.split('/')[-1]}{'_plus' if plus else ''}"


def kept_mask_from_model(model: PreTrainedModel, context_length: int) -> torch.Tensor:
    """(num_layers, num_kv_heads, T) bool of positions surviving KVzip's (fake) eviction; batch size 1."""
    layers = model.model.layers
    kept = torch.ones(
        len(layers), model.config.num_key_value_heads, context_length, dtype=torch.bool, device=model.device
    )
    for i, layer in enumerate(layers):
        indices = getattr(layer.self_attn, "masked_key_indices", None)
        if indices is not None:
            _, heads, positions = indices
            kept[i, heads.to(kept.device), positions.to(kept.device)] = False
    return kept


def move_adapters_to_base_device(model: PreTrainedModel):
    """``load_adapter`` may leave LoRA weights on CPU when the model was moved with ``.to(device)``."""
    for module in model.modules():
        base = getattr(module, "base_layer", None)
        if base is not None and hasattr(module, "lora_A"):
            device = base.weight.device
            module.lora_A.to(device)
            module.lora_B.to(device)


def prepare_press(press, model: PreTrainedModel):
    """Load a RestoreKV-style press' adapter up front and put it on the model's device."""
    if isinstance(press, RestoreKVPress):
        press.post_init_from_model(model)
        move_adapters_to_base_device(model)


def expand_restore_embeddings(emb: torch.Tensor, slots_per_region: int, num_global: int, noise: float = 0.01) -> torch.Tensor:
    """Grow a PRGF v1 layout [R locals, 1 global] into [R*k locals (k per region), g globals].

    Each region's slot is copied k times and the global slot g times; copies get small noise (relative to
    the embedding scale) so they can specialise.
    """
    n_regions = emb.shape[0] - 1
    local = emb[:n_regions].repeat_interleave(slots_per_region, dim=0)
    glob = emb[-1:].repeat(num_global, 1)
    out = torch.cat([local, glob]).clone()
    first = torch.zeros(out.shape[0], dtype=torch.bool)
    first[torch.arange(n_regions) * slots_per_region] = True
    first[n_regions * slots_per_region] = True
    out[~first] += noise * emb.std() * torch.randn_like(out[~first])
    return out


@dataclass
class PartitionedRestoreKVPress(RestoreKVPress):
    """RestoreKV with Partitioned Restore and Global Fusion.

    The n restore tokens are split into n-1 local slots R_1..R_{n-1} and one global slot G (last).
    The context is cut into n-1 contiguous regions P_j. In the single LoRA-adapted restore pass,
    R_j reads the kept cache plus the evicted positions of P_j (and itself); G reads the full context
    plus all local slots. Eviction is decided first (same KVzip scorer and budget-matched ratio as
    RestoreKV), so the final cache is kept KV + n restore slots under the same total budget.

    Parameters
    ----------
    adapter : str, optional
        Local directory or HF repo with a PEFT LoRA adapter and ``restore_embeddings.safetensors``.
        Defaults to the official RestoreKV checkpoint of the model.
    mask_mode : {"prgf", "causal"}
        "causal" gives the original RestoreKV access pattern (useful for controls / ablations).
    exchange_from : float, optional
        PRGF v2: from layer floor(exchange_from * L) on, local slots also read the global slots.
    slots_per_region, num_global : int
        Slot layout: k chained local slots per region and g global slots (v1: k = g = 1).
    selection_only : bool
        Stop after eviction selection and expose ``kept_mask`` (used by the trainer).
    """

    adapter: str | None = None
    mask_mode: MaskMode = "prgf"
    exchange_from: float | None = None
    slots_per_region: int = 1
    num_global: int = 1
    transport: bool = False  # write the slots by conserving transport of the evicted KV (see prgf/transport.py)
    transport_iters: int = 2
    transport_tau: float = 0.1
    transport_lambda_v: float = 1.0
    transport_mass_scale: float = 1.0  # beta in logit + beta*log(mass); 1 = exact conserving memory, 0 = ablation
    query_moment: str | None = None  # safetensors file with G (L, H_kv, d, d); None -> identity
    selection_only: bool = False
    drop_slots_at_decode: bool = False  # diagnostics only: evict the restore slots again after building them
    kept_mask: torch.Tensor | None = field(init=False, default=None, repr=False)
    _query_moment: torch.Tensor | None = field(init=False, default=None, repr=False)
    scores: torch.Tensor | None = field(init=False, default=None, repr=False)  # KVzip scores of the last context

    @property
    def adapter_name(self) -> str:
        if self.adapter is None:
            return super().adapter_name
        return "prgf_" + hashlib.md5(self.adapter.encode()).hexdigest()[:10]

    def post_init_from_model(self, model: PreTrainedModel):
        adapter = self.adapter or official_adapter(model, self.kvzip_plus_normalization)
        if adapter == self.restore_model_name:
            return
        embeddings = load_file(resolve_file(adapter, EMBEDDINGS_FILE))["restore_embeddings"]
        self.restore_embeddings = embeddings.to(model.device, dtype=model.dtype)
        if self.adapter_name not in getattr(model, "peft_config", {}):
            model.load_adapter(adapter, adapter_name=self.adapter_name)
            move_adapters_to_base_device(model)
        model.disable_adapters()
        self.restore_model_name = adapter

    def compress_post(self, model: PreTrainedModel):
        # 1) Eviction first: KVzip keeps B - n*L*H pairs (budget-matched, identical to RestoreKV).
        self.scores = self.score_val.clone() if self.selection_only else None  # before sinks are overwritten
        requested_ratio = self.compression_ratio
        if self.context_length > 0:
            self.compression_ratio = min(1.0, requested_ratio + self.reserved_per_head / self.context_length)
        try:
            KVzipPress.compress_post(self, model)
        finally:
            self.compression_ratio = requested_ratio
        self.kept_mask = kept_mask_from_model(model, self.context_length)
        if self.selection_only:  # trainer: keep the full cache intact, no fake eviction at decode time
            for layer in model.model.layers:
                layer.self_attn.masked_key_indices = None
            return
        # 2) Restore pass over the still-complete cache, masked per (layer, KV head).
        self.append_restore_tokens(model)
        if self.transport:
            self._transport_write(model)
        if self.drop_slots_at_decode:
            self._mask_restore_slots(model)

    @property
    def reserved_per_head(self) -> int:
        """KV pairs per (layer, head) paid for the restore memory: n slots (+1 pair holding the n mass scalars)."""
        return self.num_restore_tokens + int(self.transport)

    @property
    def transport_config(self) -> TransportConfig:
        return TransportConfig(self.transport_iters, self.transport_tau, self.transport_lambda_v)

    def query_moment_for(self, model: PreTrainedModel) -> torch.Tensor:
        if self._query_moment is None:
            L, H = model.config.num_hidden_layers, model.config.num_key_value_heads
            d = model.config.head_dim
            if self.query_moment is None:
                G = torch.eye(d).expand(L, H, d, d)
            else:
                G = load_file(self.query_moment)["G"]
            self._query_moment = G.to(model.device, torch.float32)
        return self._query_moment

    @torch.inference_mode()  # the cache tensors were created under inference mode
    def _transport_write(self, model: PreTrainedModel):
        cache, T, n = self._cache, self.context_length, self.num_restore_tokens
        n_regions = (n - self.num_global) // self.slots_per_region
        scope = slot_scope(n_regions, self.slots_per_region, self.num_global, T, model.device)
        G = self.query_moment_for(model)
        for i, (layer, cache_layer) in enumerate(zip(model.model.layers, cache.layers)):
            keys, values = cache_layer.keys[0], cache_layer.values[0]  # (H, T + n, d), post-RoPE
            mu, nu, log_mass = transport_write(
                keys[:, :T], values[:, :T], keys[:, T : T + n], values[:, T : T + n], ~self.kept_mask[i], scope, G[i],
                self.transport_config,
            )
            keys[:, T : T + n], values[:, T : T + n] = mu, nu
            log_mass = torch.where(torch.isinf(log_mass), log_mass, self.transport_mass_scale * log_mass)
            layer.self_attn.prgf_slot_bias = (T, log_mass)

    def _mask_restore_slots(self, model: PreTrainedModel):
        n, T = self.num_restore_tokens, self.context_length
        for layer in model.model.layers:
            module = layer.self_attn
            batch, heads, positions = module.masked_key_indices
            h = torch.arange(model.config.num_key_value_heads).repeat_interleave(n)
            pos = torch.arange(T, T + n).repeat(model.config.num_key_value_heads)
            module.masked_key_indices = (
                torch.cat([batch, torch.zeros_like(h).to(batch.device)]),
                torch.cat([heads, h.to(heads.device)]),
                torch.cat([positions, pos.to(positions.device)]),
            )

    def append_restore_tokens(self, model: PreTrainedModel):
        if self.mask_mode == "causal":
            return super().append_restore_tokens(model)  # exact original RestoreKV code path
        assert model.config._attn_implementation == "sdpa", "PRGF masks are implemented for SDPA attention"
        num_groups = model.config.num_attention_heads // model.config.num_key_value_heads
        provider = restore_pass_provider(
            self.kept_mask, self.num_restore_tokens, num_groups, self.mask_mode,
            exchange_layer(model.config.num_hidden_layers, self.exchange_from), self.slots_per_region, self.num_global,
        )
        with attention_bias(provider):
            super().append_restore_tokens(model)

    def select_from_scores(self, model: PreTrainedModel, scores: torch.Tensor) -> torch.Tensor:
        """Eviction selection from precomputed KVzip scores (L, 1, H, T) — same code path as compress_post.

        KVzip scores depend only on the context, so the trainer caches them instead of re-running the
        chunked reconstruction passes for every run and budget.
        """
        assert self.selection_only
        self.score_val = scores.to(model.device, model.dtype).clone()
        self.context_length = scores.shape[-1]
        try:
            self.compress_post(model)
        finally:
            self._reset_internal_parameters()
        return self.kept_mask
