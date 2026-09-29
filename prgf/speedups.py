"""Pure-overhead removals shared by every KVzip-based method (KVzip, RestoreKV, PRGF).

They change no numerics (outputs are bit-identical), only wall time, and are applied identically to
all methods in training and evaluation, so comparisons stay fair:

* KVzipPress.__call__ reloads the tokenizer from disk for every context (~1 s for Qwen3): cache it.
* KVzip stores the evicted (batch, head, position) indices on CPU, so every decoding step copies them
  to the GPU once per layer: move them to the GPU once, right after compression.
* kvpress' fake-key search synchronizes with the host at every iteration: run the same iterates on
  device and synchronize only every few iterations.
"""

from functools import lru_cache

import kvpress.attention_patch as attention_patch
import kvpress.presses.kvzip_press as kvzip_press
import torch
from transformers import AutoTokenizer

_original_search_hyperplane = attention_patch.search_hyperplane
_original_compress_post = kvzip_press.KVzipPress.compress_post


class _CachedAutoTokenizer:
    @staticmethod
    @lru_cache(maxsize=8)
    def from_pretrained(name_or_path, *args, **kwargs):
        return AutoTokenizer.from_pretrained(name_or_path, *args, **kwargs)


def search_hyperplane(X: torch.Tensor, max_iter: int = 1000, check_every: int = 4) -> torch.Tensor:
    """Same iterates and result as ``kvpress.attention_patch.search_hyperplane``.

    Once every query row satisfies <x, Y> <= 0, Y is frozen on device, so checking the stopping
    condition only every ``check_every`` iterations returns exactly the original Y.
    """
    output_dtype = X.dtype
    if output_dtype == torch.float16:
        X = X.float()
    Y = X.mean(1)
    done = torch.zeros((), dtype=torch.bool, device=X.device)
    for i in range(max_iter):
        mask = torch.bmm(X, Y.unsqueeze(-1)) <= 0
        done = done | ~mask.any()
        Y = torch.where(done, Y, Y + (X * mask).sum(1) / mask.sum(1).clamp(min=1))
        if (i + 1) % check_every == 0 and done.item():
            break
    else:
        if not done.item():
            raise ValueError("Could not find fake keys such that for every query q, exp(<q, k>) = 0")
    K = -1e5 * Y / Y.norm(dim=-1, keepdim=True) ** 2
    if output_dtype == torch.float16:
        scale = (torch.finfo(output_dtype).max / K.abs().amax(dim=-1, keepdim=True)).clamp(max=1)
        K = (K * scale).to(output_dtype)
    return K


def _compress_post_indices_on_device(self, model):
    _original_compress_post(self, model)
    for layer in model.model.layers:
        module = layer.self_attn
        indices = getattr(module, "masked_key_indices", None)
        if indices is not None:
            device = module.o_proj.weight.device
            module.masked_key_indices = tuple(t.to(device, non_blocking=True) for t in indices)


def enable():
    kvzip_press.AutoTokenizer = _CachedAutoTokenizer
    kvzip_press.KVzipPress.compress_post = _compress_post_indices_on_device
    attention_patch.search_hyperplane = search_hyperplane
