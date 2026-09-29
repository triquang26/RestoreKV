"""RULER-4K evaluation with kvpress' own pipeline and scorer.

Protocol (identical for every method):
  * dataset ``simonjegou/ruler`` / 4096, the one used by the KVPress leaderboard;
  * fixed split: ``dev`` = first ``DEV_PER_TASK`` rows of each task after a seeded shuffle (used for
    iterating), ``test`` = the remaining rows (only used for final numbers), ``test:K`` / ``dev:K`` = the
    first K rows of each task of that split (same shuffle; cheaper screening sets), ``all`` = everything;
  * rows grouped by context and answered by ``pipe(context, questions=..., answer_prefix=...)`` exactly as
    kvpress/evaluation/evaluate.py does; greedy decoding with the task's max_new_tokens;
  * SDPA attention and bf16 for all methods; score = kvpress' RULER string-match metric, averaged over tasks.
"""

import ast
import contextlib
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
    split, _, tasks = split.partition("@")  # e.g. "test:50@niah_single_2,niah_multivalue" (filter AFTER splitting)
    split, _, per_task = split.partition(":")
    assert split in ("dev", "test"), split
    k = int(per_task) if per_task else None
    rng = np.random.default_rng(SPLIT_SEED)
    rows = []
    for _, g in df.groupby("task", sort=True):
        order = rng.permutation(g.index.to_numpy())
        rows += list((order[:DEV_PER_TASK] if split == "dev" else order[DEV_PER_TASK:])[:k])
    out = df.loc[sorted(rows)]
    return out[out["task"].isin(tasks.split(","))] if tasks else out


def make_press(spec: dict, compression_ratio: float):
    """spec: {"method": "no_press" | "kvzip" | "restorekv" | "prgf", "adapter": str|None, "mask_mode": str, "plus": bool}."""
    from kvpress import KVzipPress, RestoreKVPress

    from prgf.press import PartitionedRestoreKVPress

    method, plus = spec["method"], spec.get("plus", False)
    if method == "no_press":
        return None
    extra = {k: spec[k] for k in ("chunk_size",) if k in spec}  # diagnostics
    if method == "kvzip":
        return KVzipPress(compression_ratio=compression_ratio, kvzip_plus_normalization=plus, **extra)
    if method == "restorekv" and spec.get("adapter") is None:  # the official kvpress implementation, untouched
        return RestoreKVPress(compression_ratio=compression_ratio, kvzip_plus_normalization=plus)
    mode = spec.get("mask_mode", "prgf" if method == "prgf" else "causal")
    return PartitionedRestoreKVPress(
        compression_ratio=compression_ratio, kvzip_plus_normalization=plus, adapter=spec.get("adapter"), mask_mode=mode,
        exchange_from=spec.get("exchange_from"), drop_slots_at_decode=spec.get("drop_slots_at_decode", False),
        slots_per_region=spec.get("slots_per_region", 1), num_global=spec.get("num_global", 1),
        transport=spec.get("transport", False), query_moment=spec.get("query_moment"),
        transport_iters=spec.get("transport_iters", 2), transport_tau=spec.get("transport_tau", 0.1),
        transport_lambda_v=spec.get("transport_lambda_v", 1.0), **extra,
    )


def run_rows(pipe, press, df: pd.DataFrame) -> tuple[pd.Series, float]:
    import torch

    from prgf.press import prepare_press

    prepare_press(press, pipe.model)
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


def profile(pipe, press, df: pd.DataFrame, warmup: int = 2) -> dict:
    """Per-sample wall time split into prefill, compression (scoring + restore pass) and decoding.

    Mirrors KVPressTextGenerationPipeline._forward for one question per context.
    """
    import torch
    from kvpress import RestoreKVPress
    from transformers import DynamicCache

    from prgf.press import prepare_press

    model = pipe.model
    prepare_press(press, model)
    rows = {"prefill_s": [], "compress_s": [], "decode_s": [], "new_tokens": []}

    def timed(fn):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        return out, time.perf_counter() - t0

    with torch.inference_mode():
        for i, (_, row) in enumerate(df.iterrows()):
            inputs = pipe.preprocess(row["context"], [row["question"]], row["answer_prefix"], 10**9)
            ctx = inputs["context_ids"].to(model.device)
            _, prefill = timed(lambda: model.model(input_ids=ctx, past_key_values=DynamicCache()))
            cache = DynamicCache()

            def compress():
                with press(model) if press is not None else contextlib.nullcontext():
                    model.model(input_ids=ctx, past_key_values=cache)

            _, total = timed(compress)
            length = cache.get_seq_length() if isinstance(press, RestoreKVPress) else ctx.shape[1]
            answer, decode = timed(lambda: pipe.generate_answer(
                inputs["questions_ids"][0].to(model.device), cache, length, int(row["max_new_tokens"])))
            if i >= warmup:
                rows["prefill_s"].append(prefill)
                rows["compress_s"].append(total - prefill)
                rows["decode_s"].append(decode)
                rows["new_tokens"].append(len(pipe.tokenizer.encode(answer)))
    out = {k: float(np.mean(v)) for k, v in rows.items()}
    out["decode_ms_per_token"] = 1000 * out["decode_s"] / max(out["new_tokens"], 1)
    out["n"] = len(rows["prefill_s"])
    return out
