"""Run RestoreKV (via NVIDIA KVPress) on a Modal A100 GPU.

Compares three settings on a synthetic long-context multi-fact retrieval task:
  * full cache (no eviction)
  * KVzip          (query-agnostic eviction baseline)
  * RestoreKV      (KVzip + learned restore tokens, same total KV budget)

Usage:
    modal run modal_app.py                                  # defaults: Qwen3-8B, ~8K ctx, 5% budget
    modal run modal_app.py --compression-ratio 0.9 --n-facts 20
    modal run modal_app.py --plus                           # KVzip+ / RestoreKV+ variant
"""

import modal

MODEL_CACHE = "/cache"

app = modal.App("restorekv")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    # RestoreKVPress is only on KVPress main, not (necessarily) on the PyPI release.
    .pip_install("git+https://github.com/NVIDIA/kvpress.git", "hf_transfer")
    .env({"HF_HOME": MODEL_CACHE, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Persist downloaded model weights / LoRA adapters across runs.
cache_vol = modal.Volume.from_name("restorekv-hf-cache", create_if_missing=True)


def build_task(tokenizer, n_facts: int, target_tokens: int, seed: int = 0):
    """Long filler context with `n_facts` codes hidden at random positions.

    Returns (context, questions, answers). Questions are asked *after* compression,
    which is exactly the query-agnostic setting RestoreKV targets.
    """
    import random

    rng = random.Random(seed)
    filler = (
        "The committee reviewed the quarterly logistics report and noted steady progress "
        "across warehouses, although several routes still required additional scheduling. "
    )
    cities = ["Lisbon", "Osaka", "Nairobi", "Quito", "Tallinn", "Hanoi", "Perth", "Oslo", "Lima", "Cairo",
              "Bergen", "Dakar", "Kyoto", "Sofia", "Tunis", "Accra", "Riga", "Hobart", "Zagreb", "Malmo"]
    rng.shuffle(cities)
    cities = cities[:n_facts]
    codes = {c: str(rng.randint(10000, 99999)) for c in cities}

    n_filler = max(1, target_tokens // len(tokenizer.encode(filler)))
    slots = sorted(rng.sample(range(n_filler), n_facts))
    parts, fact_iter = [], iter(cities)
    slot_set = set(slots)
    for i in range(n_filler):
        parts.append(filler)
        if i in slot_set:
            c = next(fact_iter)
            parts.append(f"IMPORTANT: The access code for the {c} depot is {codes[c]}. ")
    context = "".join(parts)
    questions = [f"What is the access code for the {c} depot? Answer with the number only." for c in cities]
    answers = [codes[c] for c in cities]
    return context, questions, answers


def run_setting(pipe, name, press, context, questions, answers):
    import time

    import torch

    torch.cuda.synchronize()
    t0 = time.time()
    out = pipe(context, questions=questions, press=press, max_new_tokens=16)["answers"]
    torch.cuda.synchronize()
    dt = time.time() - t0
    correct = sum(a in o for o, a in zip(out, answers))
    print(f"[{name:<22}] acc={correct}/{len(answers)} ({100 * correct / len(answers):.0f}%)  time={dt:.1f}s")
    for q, o, a in list(zip(questions, out, answers))[:3]:
        print(f"    gold={a}  pred={o.strip()[:40]!r}")
    return {"name": name, "correct": correct, "total": len(answers), "seconds": dt}


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={MODEL_CACHE: cache_vol},
    timeout=60 * 60,
)
def run_demo(
    model: str = "Qwen/Qwen3-8B",
    compression_ratio: float = 0.95,
    context_tokens: int = 8000,
    n_facts: int = 10,
    plus: bool = False,
):
    import torch
    from kvpress import KVzipPress, RestoreKVPress  # noqa: F401  (registers the pipeline too)
    from transformers import AutoTokenizer, pipeline

    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    tok = AutoTokenizer.from_pretrained(model)
    context, questions, answers = build_task(tok, n_facts, context_tokens)
    print(f"context tokens: {len(tok.encode(context))}, questions: {len(questions)}")

    pipe = pipeline(
        "kv-press-text-generation",
        model=model,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        model_kwargs={"attn_implementation": "sdpa"},
    )

    restore_press = RestoreKVPress(compression_ratio=compression_ratio, kvzip_plus_normalization=plus)
    # PEFT loads the LoRA adapter on CPU; load it now and move everything to the GPU.
    restore_press.post_init_from_model(pipe.model)
    pipe.model.to("cuda:0")

    results = [
        run_setting(pipe, "full cache", None, context, questions, answers),
        run_setting(
            pipe, f"KVzip{'+' if plus else ''} @{compression_ratio}",
            KVzipPress(compression_ratio=compression_ratio, kvzip_plus_normalization=plus),
            context, questions, answers,
        ),
        run_setting(
            pipe, f"RestoreKV{'+' if plus else ''} @{compression_ratio}",
            restore_press,
            context, questions, answers,
        ),
    ]
    cache_vol.commit()
    return results


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen3-8B",
    compression_ratio: float = 0.95,
    context_tokens: int = 8000,
    n_facts: int = 10,
    plus: bool = False,
):
    results = run_demo.remote(model, compression_ratio, context_tokens, n_facts, plus)
    print("\n=== Summary ===")
    for r in results:
        print(f"{r['name']:<24} {r['correct']}/{r['total']}  {r['seconds']:.1f}s")
