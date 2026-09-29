"""Self-distillation of restore embeddings + LoRA under a given restore-pass mask.

One step = one context (as in RestoreKV):
  1. prefill the context (base model, no grad) and run KVzip scoring/selection for a budget
     ratio sampled from U(budget_min, budget_max); the restore slots are paid for out of the budget;
  2. restore pass: n restore tokens, LoRA on, attention restricted by the mask (``prgf`` or ``causal``);
  3. teacher: question + teacher answer over the full cache (no grad);
     student: the same tokens over kept cache + restore slots (LoRA off, grad flows into the slots);
  4. loss = symmetric KL between teacher and student next-token distributions on the answer tokens.
The backbone and the evictor are frozen.
"""

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from prgf import speedups
from prgf.data import kvpress_inputs
from prgf.masking import MaskMode, attention_bias, exchange_layer, restore_pass_provider, student_pass_provider
from prgf.press import EMBEDDINGS_FILE, PartitionedRestoreKVPress


@dataclass
class TrainConfig:
    data_path: str
    output_dir: str
    model: str = "Qwen/Qwen3-8B"
    init_adapter: str | None = None  # None -> official RestoreKV checkpoint of `model`
    mask_mode: MaskMode = "prgf"
    exchange_from: float | None = None  # PRGF v2: local slots read G from layer floor(exchange_from * L)
    steps: int = 2000
    lr: float = 1e-4
    warmup_steps: int = 50
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    budget_min: float = 0.025
    budget_max: float = 0.25
    max_qa_per_step: int = 5
    max_answer_tokens: int = 512
    seed: int = 0
    resume_from: str | None = None  # checkpoint dir with trainer_state.pt: continue that run exactly
    score_cache_dir: str | None = None  # per-context KVzip scores, filled on first use and reused by later runs
    log_every: int = 25
    save_every: int = 500


def _cache_from(layers_kv: list[tuple[torch.Tensor, torch.Tensor]]) -> DynamicCache:
    cache = DynamicCache()
    for i, (k, v) in enumerate(layers_kv):
        cache.update(k, v, i)
    return cache


class PhaseTimer:
    """Accumulates wall time per training phase (CUDA-synchronized at phase boundaries)."""

    def __init__(self):
        self.totals: dict[str, float] = {}
        self._last = None

    def start(self):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self._last = time.perf_counter()

    def lap(self, name: str):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        now = time.perf_counter()
        self.totals[name] = self.totals.get(name, 0.0) + now - self._last
        self._last = now

    def summary(self, steps: int) -> str:
        out = " ".join(f"{k}={1000 * v / steps:.0f}ms" for k, v in self.totals.items())
        self.totals = {}
        return out


def symmetric_kl(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    lt, ls = F.log_softmax(teacher_logits.float(), -1), F.log_softmax(student_logits.float(), -1)
    kl_ts = (lt.exp() * (lt - ls)).sum(-1)
    kl_st = (ls.exp() * (ls - lt)).sum(-1)
    return 0.5 * (kl_ts + kl_st).mean()


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        speedups.enable()
        self.rng = random.Random(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.tok = AutoTokenizer.from_pretrained(cfg.model)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model, dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        ).eval()  # eval(): no dropout; gradients still flow
        self.press = PartitionedRestoreKVPress(
            adapter=cfg.resume_from or cfg.init_adapter, mask_mode=cfg.mask_mode, selection_only=True
        )
        self.press.post_init_from_model(self.model)
        self.adapter = self.press.adapter_name
        self.num_groups = self.model.config.num_attention_heads // self.model.config.num_key_value_heads

        for p in self.model.parameters():
            p.requires_grad_(False)
        self.lora = [p for n, p in self.model.named_parameters() if f".{self.adapter}." in n and "lora_" in n]
        assert self.lora, "no LoRA parameters found"
        for p in self.lora:  # fp32 master weights (PEFT casts activations to the adapter dtype)
            p.data = p.data.float()
            p.requires_grad_(True)
        self.embeddings = torch.nn.Parameter(self.press.restore_embeddings.float().clone())
        self.num_restore = self.embeddings.shape[0]

        params = [self.embeddings, *self.lora]
        self.optim = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.999), weight_decay=cfg.weight_decay)
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.optim, self._lr_lambda)
        n_params = sum(p.numel() for p in params)
        print(f"trainable params: {n_params / 1e6:.2f}M | adapter {self.adapter} | mask {cfg.mask_mode}")

        self.timer = PhaseTimer()
        with open(cfg.data_path) as f:
            self.data = [json.loads(line) for line in f]

    def _lr_lambda(self, step):
        if step < self.cfg.warmup_steps:
            return (step + 1) / self.cfg.warmup_steps
        progress = (step - self.cfg.warmup_steps) / max(1, self.cfg.steps - self.cfg.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    def _adapters(self, enabled: bool):
        if enabled:
            self.model.set_adapter(self.adapter)
            self.model.enable_adapters()
        else:
            self.model.disable_adapters()
        # PEFT toggles requires_grad of adapter weights; a leaf with requires_grad=False at backward
        # time silently drops its gradient, so keep them trainable either way.
        for p in self.lora:
            p.requires_grad_(True)

    def _score_path(self, sample) -> str | None:
        if self.cfg.score_cache_dir is None:
            return None
        return os.path.join(self.cfg.score_cache_dir, f"{sample['id']}.safetensors")

    def _load_scores(self, sample):
        path = self._score_path(sample)
        if path is None or not os.path.exists(path):
            return None
        return load_file(path, device=str(self.model.device))["scores"]

    def _save_scores(self, sample, scores):
        path = self._score_path(sample)
        if path is None:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_file({"scores": scores.contiguous().cpu()}, path + ".tmp")
        os.replace(path + ".tmp", path)  # atomic: concurrent runs never read a partial file

    def _batch(self, sample):
        ctx_ids, q_ids = kvpress_inputs(self.tok, sample["context"], sample["questions"])
        pairs = [(q, a[: self.cfg.max_answer_tokens]) for q, a in zip(q_ids, sample["answer_ids"]) if a]
        pairs = self.rng.sample(pairs, min(len(pairs), self.cfg.max_qa_per_step))
        length = max(len(q) + len(a) for q, a in pairs)
        pad = self.tok.pad_token_id or 0
        ids = torch.full((len(pairs), length), pad, dtype=torch.long)
        rows, cols = [], []
        for b, (q, a) in enumerate(pairs):
            ids[b, : len(q) + len(a)] = torch.tensor(q + a)
            rows += [b] * len(a)  # logits at position p predict token p+1: answer tokens are predicted
            cols += range(len(q) - 1, len(q) - 1 + len(a))  # from the last question token onwards
        dev = self.model.device
        return torch.tensor([ctx_ids], device=dev), ids.to(dev), (torch.tensor(rows, device=dev), torch.tensor(cols, device=dev))

    def _logits(self, qa_ids, cache, start, where):
        pos = torch.arange(start, start + qa_ids.shape[1], device=qa_ids.device)[None].expand(qa_ids.shape[0], -1)
        hidden = self.model.model(input_ids=qa_ids, past_key_values=cache, position_ids=pos).last_hidden_state
        return self.model.lm_head(hidden[where])

    def step(self, sample):
        model, n = self.model, self.num_restore
        self.timer.start()
        ctx_ids, qa_ids, where = self._batch(sample)
        batch = qa_ids.shape[0]

        # 1) prefill + KVzip selection (reserves n*L*H pairs of the budget for the restore slots)
        self.press.compression_ratio = 1 - self.rng.uniform(self.cfg.budget_min, self.cfg.budget_max)
        cache, cached = DynamicCache(), self._load_scores(sample)
        with torch.no_grad():
            if cached is None:
                with self.press(model):
                    model.model(input_ids=ctx_ids, past_key_values=cache)
                self._save_scores(sample, self.press.scores)
                kept = self.press.kept_mask
            else:
                model.model(input_ids=ctx_ids, past_key_values=cache)
                kept = self.press.select_from_scores(model, cached)
        T = ctx_ids.shape[1]
        ctx_kv = [(layer.keys, layer.values) for layer in cache.layers]
        self.timer.lap("select")

        # 2) teacher over the full cache
        with torch.no_grad():
            full = _cache_from([(k.expand(batch, -1, -1, -1), v.expand(batch, -1, -1, -1)) for k, v in ctx_kv])
            teacher = self._logits(qa_ids, full, T, where)
            del full
        self.timer.lap("teacher")

        # 3) restore pass (LoRA on) with the method's mask
        pos = torch.arange(T, T + n, device=model.device)
        restore_cache = _cache_from(ctx_kv)
        self._adapters(True)
        try:
            provider = restore_pass_provider(
                kept, n, self.num_groups, self.cfg.mask_mode,
                exchange_layer(model.config.num_hidden_layers, self.cfg.exchange_from),
            )
            with attention_bias(provider):
                model.model(
                    inputs_embeds=self.embeddings.to(model.dtype)[None],
                    past_key_values=restore_cache, position_ids=pos[None], cache_position=pos, use_cache=True,
                )
        finally:
            self._adapters(False)
        slots = [(layer.keys[:, :, T:], layer.values[:, :, T:]) for layer in restore_cache.layers]
        self.timer.lap("restore")

        # 4) student over kept cache + restore slots (evicted positions are hard-masked)
        student_cache = _cache_from([
            (torch.cat([k, sk], 2).expand(batch, -1, -1, -1), torch.cat([v, sv], 2).expand(batch, -1, -1, -1))
            for (k, v), (sk, sv) in zip(ctx_kv, slots)
        ])
        with attention_bias(student_pass_provider(kept, n, self.num_groups)):
            student = self._logits(qa_ids, student_cache, T + n, where)

        loss = symmetric_kl(teacher, student)
        self.timer.lap("student_fwd")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([self.embeddings, *self.lora], self.cfg.max_grad_norm)
        self.optim.step()
        self.sched.step()
        self.optim.zero_grad(set_to_none=True)
        self.timer.lap("backward_optim")
        return loss.item(), grad_norm.item(), 1 - self.press.compression_ratio

    def save(self, path, step: int, order: list[int]):
        os.makedirs(path, exist_ok=True)
        self.model.set_adapter(self.adapter)
        self.model.save_pretrained(path)  # adapter-only save with transformers' PEFT integration
        save_file({"restore_embeddings": self.embeddings.detach().to(torch.bfloat16).cpu()}, os.path.join(path, EMBEDDINGS_FILE))
        with open(os.path.join(path, "train_config.json"), "w") as f:
            json.dump(asdict(self.cfg), f, indent=2)
        state = {
            "step": step, "order": order, "optim": self.optim.state_dict(), "sched": self.sched.state_dict(),
            "rng": self.rng.getstate(), "torch_rng": torch.get_rng_state(),
            # fp32 master weights: the saved adapter is re-loaded in the model dtype (bf16)
            "fp32_embeddings": self.embeddings.detach().cpu(), "fp32_lora": [p.detach().cpu() for p in self.lora],
        }
        torch.save(state, os.path.join(path, "trainer_state.pt"))

    def _resume(self) -> tuple[int, list[int]]:
        state = torch.load(os.path.join(self.cfg.resume_from, "trainer_state.pt"), weights_only=False)
        self.optim.load_state_dict(state["optim"])
        self.sched.load_state_dict(state["sched"])
        self.rng.setstate(state["rng"])
        torch.set_rng_state(state["torch_rng"])
        with torch.no_grad():  # bf16 copies were loaded through the press; restore the fp32 master weights
            self.embeddings.copy_(state["fp32_embeddings"].to(self.embeddings.device))
            for p, saved in zip(self.lora, state["fp32_lora"], strict=True):
                p.copy_(saved.to(p.device))
        print(f"resumed from {self.cfg.resume_from} at step {state['step']}")
        return state["step"], state["order"]

    def train(self, on_save=None):
        start, order = self._resume() if self.cfg.resume_from else (0, [])
        t0, window = time.time(), []
        for step in range(start, self.cfg.steps):
            if not order:
                order = list(range(len(self.data)))
                self.rng.shuffle(order)
            loss, gnorm, budget = self.step(self.data[order.pop()])
            window.append(loss)
            if (step + 1) % self.cfg.log_every == 0:
                rate = (step + 1 - start) / (time.time() - t0)
                print(
                    f"step {step + 1}/{self.cfg.steps} loss {sum(window) / len(window):.4f} gnorm {gnorm:.3f} "
                    f"budget {budget:.3f} lr {self.sched.get_last_lr()[0]:.2e} {rate:.2f} it/s | "
                    f"{self.timer.summary(len(window))}",
                    flush=True,
                )
                window = []
            if (step + 1) % self.cfg.save_every == 0 or step + 1 == self.cfg.steps:
                path = os.path.join(self.cfg.output_dir, f"step{step + 1}" if step + 1 < self.cfg.steps else "final")
                self.save(path, step + 1, order)
                if on_save:
                    on_save()


def latest_checkpoint(output_dir: str) -> str | None:
    """Most advanced resumable checkpoint of a run (``final`` excluded: that run is complete)."""
    if not os.path.isdir(output_dir):
        return None
    steps = [d for d in os.listdir(output_dir)
             if d.startswith("step") and os.path.exists(os.path.join(output_dir, d, "trainer_state.pt"))]
    return os.path.join(output_dir, max(steps, key=lambda d: int(d[4:]))) if steps else None
