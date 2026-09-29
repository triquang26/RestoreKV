"""Where does the evictor drop information? Per-context eviction profile of KVzip (budget-matched, as in PRGF).

For each context: fraction of kept KV pairs per PRGF region and per KVzip reconstruction chunk, the score
level of each chunk, and whether the tokens of each needle value survive eviction.
"""

import pandas as pd
import torch
from transformers import DynamicCache

from prgf.masking import region_ids
from prgf.press import PartitionedRestoreKVPress


def _value_token_span(tokenizer, context_text: str, value: str, prefix_len: int):
    """Token positions (in the cache) covering the first occurrence of ``value`` in the context."""
    start = context_text.find(value)
    if start < 0:
        return None
    lo = len(tokenizer.encode(context_text[:start], add_special_tokens=False))
    hi = len(tokenizer.encode(context_text[: start + len(value)], add_special_tokens=False))
    return prefix_len + max(lo - 1, 0), prefix_len + max(hi, lo + 1) + 1  # +-1 token for boundary merges


def eviction_profile(pipe, df: pd.DataFrame, compression_ratio: float, num_regions: int = 7, adapter=None) -> dict:
    model, tok = pipe.model, pipe.tokenizer
    press = PartitionedRestoreKVPress(compression_ratio=compression_ratio, selection_only=True, adapter=adapter)
    region_rows, value_rows = [], []
    with torch.inference_mode():
        for idx, row in df.iterrows():
            ids = pipe.preprocess(row["context"], [""], "", 10**9)["context_ids"].to(model.device)
            with press(model):
                model.model(input_ids=ids, past_key_values=DynamicCache())
            kept, scores = press.kept_mask.float(), press.scores.float()[:, 0]  # (L, H, T)
            T = kept.shape[-1]
            # chat-template tokens before the raw text (approximate at the boundary by at most one token)
            chat_prefix = max(T - len(tok.encode(row["context"], add_special_tokens=False)), 0)
            chunk2_start = chat_prefix + press.chunk_size  # KVzip reconstructs the text in chunks of chunk_size
            regions = region_ids(T, num_regions, kept.device)
            per_pos = kept.mean(dim=(0, 1))  # fraction of (layer, head) keeping each position
            for r in range(num_regions):
                region_rows.append(dict(index=idx, task=row["task"], region=r, kept=per_pos[regions == r].mean().item()))
            s = scores.mean(dim=(0, 1))  # mean KVzip score per position (sinks excluded below)
            region_rows.append(dict(index=idx, task=row["task"], region="chunk1", kept=per_pos[press.n_sink : chunk2_start].mean().item(),
                                    score=s[press.n_sink : chunk2_start].median().item()))
            if chunk2_start < T:
                region_rows.append(dict(index=idx, task=row["task"], region="chunk2", kept=per_pos[chunk2_start:].mean().item(),
                                        score=s[chunk2_start:].median().item()))
            for value in row["answer"]:
                span = _value_token_span(tok, row["context"], value, chat_prefix)
                if span is None or span[1] > T:
                    continue
                lo, hi = span
                value_rows.append(dict(
                    index=idx, task=row["task"], value=value, depth=lo / T, after_chunk_boundary=lo >= chunk2_start,
                    kept_any_head=kept[:, :, lo:hi].amax(dim=1).mean().item(),  # layers where some head keeps a value token
                    kept_frac=kept[:, :, lo:hi].mean().item(),
                ))
    return {"regions": pd.DataFrame(region_rows).to_dict("records"), "values": pd.DataFrame(value_rows).to_dict("records")}
