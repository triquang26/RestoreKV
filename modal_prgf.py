"""Modal jobs for Partitioned Restore with Global Fusion (PRGF): data, training, RULER-4K evaluation.

    modal run modal_prgf.py::build_data                         # teacher QA data -> /runs/data/train.jsonl
    modal run modal_prgf.py::train --name prgf --mask-mode prgf  # -> /runs/ckpt/<name>/final
    modal run modal_prgf.py::evaluate --name kvzip --spec '{"method": "kvzip"}' --ratios 0.9,0.95 --split dev
    modal run modal_prgf.py::report --split dev                 # table of everything under /runs/results
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
    .env({"HF_HOME": HF, "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("prgf")
)


# ----------------------------------------------------------------------------- data
@app.function(image=vllm_image, gpu="A100-80GB", volumes=volumes, timeout=4 * 3600)
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

    data = collect_contexts(AutoTokenizer.from_pretrained(MODEL), n_longalpaca, n_pg19, n_flan, seed=seed)
    print(f"collected {len(data)} contexts")
    out = [d for shard in generate_qa_shard.map([data[i::shards] for i in range(shards)]) for d in shard]
    os.makedirs(f"{RUNS}/data", exist_ok=True)
    with open(f"{RUNS}/data/train.jsonl", "w") as f:
        f.writelines(json.dumps(d) + "\n" for d in sorted(out, key=lambda d: d["id"]))
    runs.commit()
    n_qa = sum(len(d["answer_ids"]) for d in out)
    print(f"wrote {len(out)} contexts / {n_qa} QA pairs")


@app.local_entrypoint()
def build_data(n_longalpaca: int = 1200, n_pg19: int = 1000, n_flan: int = 500, shards: int = 4, seed: int = 0):
    build_data_remote.remote(n_longalpaca, n_pg19, n_flan, shards, seed)


# ----------------------------------------------------------------------------- training
@app.function(image=kv_image, gpu="A100-80GB", volumes=volumes, timeout=24 * 3600)
def train_remote(cfg: dict):
    from prgf.train import TrainConfig, Trainer

    Trainer(TrainConfig(**cfg)).train(on_save=runs.commit)
    runs.commit()


@app.local_entrypoint()
def train(name: str, mask_mode: str = "prgf", steps: int = 2000, lr: float = 1e-4, seed: int = 0):
    cfg = dict(
        data_path=f"{RUNS}/data/train.jsonl", output_dir=f"{RUNS}/ckpt/{name}", model=MODEL,
        mask_mode=mask_mode, steps=steps, lr=lr, seed=seed,
    )
    train_remote.remote(cfg)


# ----------------------------------------------------------------------------- evaluation
@app.cls(image=kv_image, gpu="A100", volumes=volumes, timeout=6 * 3600, max_containers=10)
class Evaluator:
    @modal.enter()
    def load(self):
        import torch
        from transformers import pipeline

        import prgf.press  # noqa: F401  (installs the masked-SDPA hook before the model runs)

        self.pipe = pipeline(
            "kv-press-text-generation", model=MODEL, device="cuda:0", dtype=torch.bfloat16,
            model_kwargs={"attn_implementation": "sdpa"},
        )
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
        res[["task", "question", "answer", "predicted_answer"]].to_json(f"{out_dir}/{r}_preds.jsonl", orient="records", lines=True)
        summary[r] = metrics["average"]
        print(f"{name} ratio={r} {split}: {metrics['average']:.2f}  ({metrics['gpu_seconds'] / len(res):.2f} GPU-s/sample)")
    runs.commit()
    return summary


@app.local_entrypoint()
def evaluate(name: str, spec: str, ratios: str = "0.8,0.9,0.95", split: str = "dev", n_shards: int = 4):
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
