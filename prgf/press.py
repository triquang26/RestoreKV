"""KVPress press for Partitioned Restore with Global Fusion (PRGF)."""

import hashlib
import os
from dataclasses import dataclass, field

import torch
from huggingface_hub import hf_hub_download
from kvpress import KVzipPress, RestoreKVPress
from safetensors.torch import load_file
from transformers import PreTrainedModel

from prgf.masking import MaskMode, attention_bias, restore_pass_provider

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
    selection_only : bool
        Stop after eviction selection and expose ``kept_mask`` (used by the trainer).
    """

    adapter: str | None = None
    mask_mode: MaskMode = "prgf"
    selection_only: bool = False
    kept_mask: torch.Tensor | None = field(init=False, default=None, repr=False)

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
        model.disable_adapters()
        self.restore_model_name = adapter

    def compress_post(self, model: PreTrainedModel):
        # 1) Eviction first: KVzip keeps B - n*L*H pairs (budget-matched, identical to RestoreKV).
        requested_ratio = self.compression_ratio
        if self.context_length > 0:
            self.compression_ratio = min(1.0, requested_ratio + self.num_restore_tokens / self.context_length)
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

    def append_restore_tokens(self, model: PreTrainedModel):
        if self.mask_mode == "causal":
            return super().append_restore_tokens(model)  # exact original RestoreKV code path
        assert model.config._attn_implementation == "sdpa", "PRGF masks are implemented for SDPA attention"
        num_groups = model.config.num_attention_heads // model.config.num_key_value_heads
        provider = restore_pass_provider(self.kept_mask, self.num_restore_tokens, num_groups, self.mask_mode)
        with attention_bias(provider):
            super().append_restore_tokens(model)
