"""Pure-overhead removals shared by every KVzip-based method (KVzip, RestoreKV, PRGF).

They change no numerics, only wall time, and are applied identically to all methods in training and
evaluation, so comparisons stay fair.

Profiled on A100 / RULER-4K (per context): prefill ~340 ms, KVzip scoring ~1.05 s, restore pass
~65 ms (RestoreKV) / ~120 ms (PRGF), decoding ~75 ms/token vs ~42 ms/token without eviction. The decode
gap comes from kvpress' fake-key search (tens of tiny kernels per layer and token); it is shared by all
methods and left untouched so results stay bit-comparable with kvpress.
"""

from functools import lru_cache

import kvpress.presses.kvzip_press as kvzip_press
from transformers import AutoTokenizer


class _CachedAutoTokenizer:
    """KVzipPress.__call__ reloads the tokenizer from disk for every context (~1 s for Qwen3)."""

    @staticmethod
    @lru_cache(maxsize=8)
    def from_pretrained(name_or_path, *args, **kwargs):
        return AutoTokenizer.from_pretrained(name_or_path, *args, **kwargs)


def enable():
    kvzip_press.AutoTokenizer = _CachedAutoTokenizer
