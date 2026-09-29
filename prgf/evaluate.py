"""RULER-4K evaluation with kvpress' own pipeline and scorer.

Protocol (identical for every method):
  * dataset ``simonjegou/ruler`` / 4096, the one used by the KVPress leaderboard;
  * fixed split: ``dev`` = first ``DEV_PER_TASK`` rows of each task after a seeded shuffle (used for
    iterating), ``test`` = the remaining rows (only used for final numbers), ``all`` = everything;
  * rows grouped by context and answered by ``pipe(context, questions=..., answer_prefix=...)`` exactly as
    kvpress/evaluation/evaluate.py does; greedy decoding with the task's max_new_tokens;
  * SDPA attention and bf16 for all methods; score = kvpress' RULER string-match metric, averaged over tasks.
"""

import ast
import time

import numpy as np
import pandas as pd

DEV_PER_TASK = 40
SPLIT_SEED = 0


def load_ruler(split: str) -> pd.DataFrame:
    from datasets import load_dataset

    df = load_dataset("simonjegou/ruler", data_dir="4096", split="test").to_pandas()
    df["answer"] = df["answer"].apply(lambda a: list(ast.literal_eval(a)) if isinstance(a, str) else list(a))
    df["max_new_tokens"] = df["max_new_tokens"].astype(int)
    if split == "all":
        return df
    rng = np.random.default_rng(SPLIT_SEED)
    dev_idx = []
    for _, g in df.groupby("task", sort=True):
        dev_idx += list(rng.permutation(g.index.to_numpy())[:DEV_PER_TASK])
    dev = df.index.isin(dev_idx)
    return df[dev] if split == "dev" else df[~dev]


def make_press(spec: dict, compression_ratio: float):
    """spec: {"method": "no_press" | "kvzip" | "restorekv" | "prgf", "adapter": str|None, "mask_mode": str, "plus": bool}."""
    from kvpress import KVzipPress, RestoreKVPress

    from prgf.press import PartitionedRestoreKVPress

    method, plus = spec["method"], spec.get("plus", False)
    if method == "no_press":
        return None
    if method == "kvzip":
        return KVzipPress(compression_ratio=compression_ratio, kvzip_plus_normalization=plus)
    if method == "restorekv" and spec.get("adapter") is None:  # the official kvpress implementation, untouched
        return RestoreKVPress(compression_ratio=compression_ratio, kvzip_plus_normalization=plus)
    mode = spec.get("mask_mode", "prgf" if method == "prgf" else "causal")
    return PartitionedRestoreKVPress(
        compression_ratio=compression_ratio, kvzip_plus_normalization=plus, adapter=spec.get("adapter"), mask_mode=mode
    )


def run_rows(pipe, press, df: pd.DataFrame) -> tuple[pd.Series, float]:
    import torch

    preds = pd.Series(index=df.index, dtype=object)
    t0 = time.time()
    with torch.inference_mode():
        for context, group in df.groupby("context", sort=False):
            out = pipe(
                context,
                questions=group["question"].tolist(),
                answer_prefix=group["answer_prefix"].iloc[0],
                press=press,
                max_new_tokens=int(group["max_new_tokens"].iloc[0]),
            )
            preds[group.index] = out["answers"]
    return preds, time.time() - t0


def score(df: pd.DataFrame) -> dict:
    from benchmarks.ruler.calculate_metrics import calculate_metrics  # kvpress/evaluation on PYTHONPATH

    per_task = {k: v["string_match"] for k, v in calculate_metrics(df.copy()).items()}
    return {"average": round(float(np.mean(list(per_task.values()))), 2), "per_task": per_task, "n": len(df)}
