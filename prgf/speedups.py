"""Pure-overhead removals shared by every KVzip-based method (KVzip, RestoreKV, PRGF).

They change no numerics, only wall time, and are applied identically to all methods in training and
evaluation, so comparisons stay fair.
"""

from functools import lru_cache

import kvpress.presses.kvzip_press as kvzip_press
from transformers import AutoTokenizer


class _CachedAutoTokenizer:
    """KVzipPress.__call__ reloads the tokenizer from disk for every context (~1s for Qwen3)."""

    @staticmethod
    @lru_cache(maxsize=8)
    def from_pretrained(name_or_path, *args, **kwargs):
        return AutoTokenizer.from_pretrained(name_or_path, *args, **kwargs)


def enable():
    kvzip_press.AutoTokenizer = _CachedAutoTokenizer
