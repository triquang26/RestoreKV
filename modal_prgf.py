"""Modal jobs for Partitioned Restore with Global Fusion (PRGF): data, training, RULER-4K evaluation.

    modal run modal_prgf.py::build_data                         # teacher QA data -> /runs/data/train.jsonl
    modal run modal_prgf.py::train --name prgf --mask-mode prgf  # -> /runs/ckpt/<name>/final
    modal run modal_prgf.py::evaluate --name kvzip --spec '{"method": "kvzip"}' --ratios 0.9,0.95 --split dev
    modal run modal_prgf.py::report --split dev                 # table of everything under /runs/results

GPU budget: every GPU function is capped (max_containers) so that one data/eval job plus one training
job never exceed 4 concurrent A100s. Run at most one evaluate job at a time.
"""

import json

import modal

KVPRESS_COMMIT = "66c5b0903b3be742bcba6b96d9796bd561e58f2d"  # NVIDIA/kvpress main, 2026-09-28
MODEL = "Qwen/Qwen3-8B"
HF, RUNS = "/cache", "/runs"

app = modal.App("restorekv-prgf")
hf_cache = modal.Volume.from_name("restorekv-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("prgf-runs", create_if_missing=True)
volumes = {HF: hf_cache, RUNS: runs}

kv_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands(
        "git clone https://github.com/NVIDIA/kvpress.git /opt/kvpress",
        f"cd /opt/kvpress && git checkout {KVPRESS_COMMIT} && pip install .",
    )
    .pip_install("datasets", "pandas")
    .env({"HF_HOME": HF, "PYTHONPATH": "/opt/kvpress/evaluation", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("prgf")
)
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm", "datasets")
    # FlashInfer's sampler JIT-compiles with nvcc, which the slim image does not ship.
    .env({"HF_HOME": HF, "TOKENIZERS_PARALLELISM": "false", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_python_source("prgf")
)


# ----------------------------------------------------------------------------- data
@app.function(image=vllm_image, gpu="A100-80GB", volumes=volumes, timeout=4 * 3600, max_containers=2)
def generate_qa_shard(items: list[dict]) -> list[dict]:
    from transformers import AutoTokenizer
    from vllm import LLM

    from prgf.data import generate_qa

    llm = LLM(MODEL, dtype="bfloat16", max_model_len=16384, enable_prefix_caching=True, gpu_memory_utilization=0.9)
    return generate_qa(llm, AutoTokenizer.from_pretrained(MODEL), items)


@app.function(image=vllm_image, volumes=volumes, cpu=8, memory=32768, timeout=6 * 3600)
def build_data_remote(n_longalpaca: int, n_pg19: int, n_flan: int, shards: int, seed: int):
    import os

    from transformers import AutoTokenizer

    from prgf.data import collect_contexts

    contexts_path = f"{RUNS}/data/contexts_{n_longalpaca}_{n_pg19}_{n_flan}_{seed}.jsonl"
    if os.path.exists(contexts_path):
        data = [json.loads(line) for line in open(contexts_path)]
    else:
        data = collect_contexts(AutoTokenizer.from_pretrained(MODEL), n_longalpaca, n_pg19, n_flan, seed=seed)
        os.makedirs(f"{RUNS}/data", exist_ok=True)
        with open(contexts_path, "w") as f:
            f.writelines(json.dumps(d) + "\n" for d in data)
        runs.commit()
    print(f"{len(data)} contexts")
    out = [d for shard in generate_qa_shard.map([data[i::shards] for i in range(shards)]) for d in shard]
    with open(f"{RUNS}/data/train.jsonl", "w") as f:
        f.writelines(json.dumps(d) + "\n" for d in sorted(out, key=lambda d: d["id"]))
    runs.commit()
    n_qa = sum(len(d["answer_ids"]) for d in out)
    print(f"wrote {len(out)} contexts / {n_qa} QA pairs")


@app.local_entrypoint()
def build_data(n_longalpaca: int = 1200, n_pg19: int = 1000, n_flan: int = 500, shards: int = 2, seed: int = 0):
    build_data_remote.remote(n_longalpaca, n_pg19, n_flan, shards, seed)


# ----------------------------------------------------------------------------- training
@app.function(image=kv_image, gpu="A100-80GB", volumes=volumes, timeout=24 * 3600, max_containers=2)
def train_remote(cfg: dict):
    from prgf.train import TrainConfig, Trainer

    trainer = Trainer(TrainConfig(**cfg))
    hf_cache.commit()
    trainer.train(on_save=runs.commit)
    runs.commit()


@app.local_entrypoint()
def train(
    name: str, mask_mode: str = "prgf", steps: int = 2000, lr: float = 1e-4, max_answer_tokens: int = 256,
    seed: int = 0, init_adapter: str = "", exchange_from: float = -1.0, warmup_steps: int = 50, save_every: int = 500,
):
    cfg = dict(
        data_path=f"{RUNS}/data/train.jsonl", output_dir=f"{RUNS}/ckpt/{name}", model=MODEL,
        mask_mode=mask_mode, steps=steps, lr=lr, max_answer_tokens=max_answer_tokens, seed=seed,
        init_adapter=init_adapter or None, exchange_from=exchange_from if exchange_from >= 0 else None,
        warmup_steps=warmup_steps, save_every=save_every, score_cache_dir=f"{RUNS}/data/kvzip_scores",
    )
    train_remote.remote(cfg)


# ----------------------------------------------------------------------------- evaluation
@app.cls(image=kv_image, gpu="A100", volumes=volumes, timeout=6 * 3600, max_containers=2)
class Evaluator:
    @modal.enter()
    def load(self):
        import torch
        from transformers import pipeline

        import prgf.press  # noqa: F401  (installs the masked-SDPA hook before the model runs)
        from prgf import speedups

        speedups.enable()

        self.pipe = pipeline(
            "kv-press-text-generation", model=MODEL, device="cuda:0", dtype=torch.bfloat16,
            model_kwargs={"attn_implementation": "sdpa"},
        )
        hf_cache.commit()  # keep downloaded weights for the next containers
        self.data = {}

    @modal.method()
    def run(self, spec: dict, ratio: float, split: str, shard: int, n_shards: int) -> dict:
        from prgf.evaluate import load_ruler, make_press, run_rows

        runs.reload()
        if split not in self.data:
            self.data[split] = load_ruler(split)
        df = self.data[split]
        contexts = df["context"].unique()[shard::n_shards]
        preds, seconds = run_rows(self.pipe, make_press(spec, ratio), df[df["context"].isin(contexts)])
        return {"preds": preds.to_dict(), "seconds": seconds}


    @modal.method()
    def profile(self, specs: dict, ratio: float, n_samples: int) -> dict:
        from prgf.evaluate import load_ruler, make_press, profile

        runs.reload()
        df = load_ruler("dev").groupby("task").head(max(1, n_samples // 13))
        return {name: profile(self.pipe, make_press(spec, ratio), df) for name, spec in specs.items()}


@app.function(image=kv_image, volumes=volumes, timeout=12 * 3600)
def evaluate_remote(name: str, spec: dict, ratios: list[float], split: str, n_shards: int) -> dict:
    import os

    from prgf.evaluate import load_ruler, score

    df = load_ruler(split)
    jobs = [(spec, r, split, s, n_shards) for r in ratios for s in range(n_shards)]
    outs = list(Evaluator().run.starmap(jobs))
    summary = {}
    for i, r in enumerate(ratios):
        chunk = outs[i * n_shards : (i + 1) * n_shards]
        res = df.copy()
        res["predicted_answer"] = None
        for o in chunk:
            for idx, p in o["preds"].items():
                res.at[int(idx), "predicted_answer"] = p
        assert res["predicted_answer"].notna().all()
        metrics = score(res)
        metrics.update(spec=spec, ratio=r, split=split, gpu_seconds=sum(o["seconds"] for o in chunk))
        out_dir = f"{RUNS}/results/{name}/{split}"
        os.makedirs(out_dir, exist_ok=True)
        with open(f"{out_dir}/{r}.json", "w") as f:
            json.dump(metrics, f, indent=2)
        res.reset_index()[["index", "task", "question", "answer", "predicted_answer"]].to_json(
            f"{out_dir}/{r}_preds.jsonl", orient="records", lines=True
        )
        summary[r] = metrics["average"]
        print(f"{name} ratio={r} {split}: {metrics['average']:.2f}  ({metrics['gpu_seconds'] / len(res):.2f} GPU-s/sample)")
    runs.commit()
    return summary


@app.local_entrypoint()
def evaluate(name: str, spec: str, ratios: str = "0.8,0.9,0.95", split: str = "dev", n_shards: int = 2):
    print(evaluate_remote.remote(name, json.loads(spec), [float(r) for r in ratios.split(",")], split, n_shards))


@app.function(image=kv_image, volumes=volumes)
def report_remote(split: str) -> str:
    import glob
    import os

    rows = {}
    for path in sorted(glob.glob(f"{RUNS}/results/*/{split}/*.json")):
        name, ratio = path.split("/")[-3], os.path.basename(path)[:-5]
        rows.setdefault(name, {})[ratio] = json.load(open(path))["average"]
    ratios = sorted({r for v in rows.values() for r in v}, key=float)
    lines = ["| method | " + " | ".join(f"cr={r}" for r in ratios) + " |", "|---" * (len(ratios) + 1) + "|"]
    for name, v in rows.items():
        lines.append(f"| {name} | " + " | ".join(f"{v[r]:.2f}" if r in v else "-" for r in ratios) + " |")
    return "\n".join(lines)


@app.local_entrypoint()
def report(split: str = "dev"):
    print(report_remote.remote(split))


@app.local_entrypoint()
def profile(specs: str, ratio: float = 0.9, n_samples: int = 39):
    """specs: JSON {name: spec}; measured sequentially on one GPU, stratified over RULER tasks."""
    for name, r in Evaluator().profile.remote(json.loads(specs), ratio, n_samples).items():
        print(f"{name:<16} prefill {r['prefill_s'] * 1e3:7.1f} ms | compress {r['compress_s'] * 1e3:7.1f} ms | "
              f"decode {r['decode_s'] * 1e3:7.1f} ms ({r['new_tokens']:.1f} tok, {r['decode_ms_per_token']:.1f} ms/tok) | n={r['n']}")


# ----------------------------------------------------------------------------- smoke test
@app.function(image=kv_image, gpu="A100", volumes=volumes, timeout=1800)
def smoke_remote(adapter: str | None = None):
    """One RULER-like query through every restore press on a real GPU (catches device/adapter issues)."""
    import torch
    from transformers import pipeline

    from prgf import speedups
    from prgf.evaluate import make_press
    from prgf.press import prepare_press

    speedups.enable()

    pipe = pipeline(
        "kv-press-text-generation", model=MODEL, device="cuda:0", dtype=torch.bfloat16,
        model_kwargs={"attn_implementation": "sdpa"},
    )
    context = "Filler text about nothing in particular. " * 150 + "The secret code is 4242. " + "More filler. " * 150
    for spec in [{"method": "restorekv"}, {"method": "restorekv", "adapter": adapter or f"higokri/RestoreKV-{MODEL.split('/')[-1]}"},
                 {"method": "prgf", "adapter": adapter}]:
        press = make_press(spec, 0.9)
        prepare_press(press, pipe.model)
        devices = {str(p.device) for n, p in pipe.model.named_parameters() if "lora_" in n}
        answer = pipe(context, question="What is the secret code?", press=press, max_new_tokens=8)["answer"]
        print(f"{spec} lora devices={devices} answer={answer!r}")


@app.local_entrypoint()
def smoke(adapter: str = ""):
    smoke_remote.remote(adapter or None)


# ----------------------------------------------------------------------------- training profiler
@app.function(image=kv_image, gpu="A100-80GB", volumes=volumes, timeout=3600)
def profile_train_remote(mask_mode: str, steps: int):
    import torch
    from torch.profiler import ProfilerActivity, profile

    from prgf.train import TrainConfig, Trainer

    trainer = Trainer(TrainConfig(
        data_path=f"{RUNS}/data/train.jsonl", output_dir="/tmp/profile", model=MODEL, mask_mode=mask_mode,
        max_answer_tokens=256, score_cache_dir=f"{RUNS}/data/kvzip_scores",
    ))
    for i in range(3):  # warm-up
        trainer.step(trainer.data[i])
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in range(3, 3 + steps):
            trainer.step(trainer.data[i])
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=60))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15, max_name_column_width=60))
    print(trainer.timer.summary(steps + 3))


@app.local_entrypoint()
def profile_train(mask_mode: str = "prgf", steps: int = 5):
    profile_train_remote.remote(mask_mode, steps)
