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
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from prgf.data import kvpress_inputs
from prgf.masking import MaskMode, attention_bias, restore_pass_provider, student_pass_provider
from prgf.press import EMBEDDINGS_FILE, PartitionedRestoreKVPress


@dataclass
class TrainConfig:
    data_path: str
    output_dir: str
    model: str = "Qwen/Qwen3-8B"
    init_adapter: str | None = None  # None -> official RestoreKV checkpoint of `model`
    mask_mode: MaskMode = "prgf"
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
    log_every: int = 25
    save_every: int = 500


def _cache_from(layers_kv: list[tuple[torch.Tensor, torch.Tensor]]) -> DynamicCache:
    cache = DynamicCache()
    for i, (k, v) in enumerate(layers_kv):
        cache.update(k, v, i)
    return cache


def symmetric_kl(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    lt, ls = F.log_softmax(teacher_logits.float(), -1), F.log_softmax(student_logits.float(), -1)
    kl_ts = (lt.exp() * (lt - ls)).sum(-1)
    kl_st = (ls.exp() * (ls - lt)).sum(-1)
    return 0.5 * (kl_ts + kl_st).mean()


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.tok = AutoTokenizer.from_pretrained(cfg.model)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model, dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        ).eval()  # eval(): no dropout; gradients still flow
        self.press = PartitionedRestoreKVPress(adapter=cfg.init_adapter, mask_mode=cfg.mask_mode, selection_only=True)
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
        ctx_ids, qa_ids, where = self._batch(sample)
        batch = qa_ids.shape[0]

        # 1) prefill + KVzip selection (reserves n*L*H pairs of the budget for the restore slots)
        self.press.compression_ratio = 1 - self.rng.uniform(self.cfg.budget_min, self.cfg.budget_max)
        cache = DynamicCache()
        with torch.no_grad(), self.press(model):
            model.model(input_ids=ctx_ids, past_key_values=cache)
        kept, T = self.press.kept_mask, ctx_ids.shape[1]
        ctx_kv = [(layer.keys, layer.values) for layer in cache.layers]

        # 2) teacher over the full cache
        with torch.no_grad():
            full = _cache_from([(k.expand(batch, -1, -1, -1), v.expand(batch, -1, -1, -1)) for k, v in ctx_kv])
            teacher = self._logits(qa_ids, full, T, where)
            del full

        # 3) restore pass (LoRA on) with the method's mask
        pos = torch.arange(T, T + n, device=model.device)
        restore_cache = _cache_from(ctx_kv)
        self._adapters(True)
        try:
            with attention_bias(restore_pass_provider(kept, n, self.num_groups, self.cfg.mask_mode)):
                model.model(
                    inputs_embeds=self.embeddings.to(model.dtype)[None],
                    past_key_values=restore_cache, position_ids=pos[None], cache_position=pos, use_cache=True,
                )
        finally:
            self._adapters(False)
        slots = [(layer.keys[:, :, T:], layer.values[:, :, T:]) for layer in restore_cache.layers]

        # 4) student over kept cache + restore slots (evicted positions are hard-masked)
        student_cache = _cache_from([
            (torch.cat([k, sk], 2).expand(batch, -1, -1, -1), torch.cat([v, sv], 2).expand(batch, -1, -1, -1))
            for (k, v), (sk, sv) in zip(ctx_kv, slots)
        ])
        with attention_bias(student_pass_provider(kept, n, self.num_groups)):
            student = self._logits(qa_ids, student_cache, T + n, where)

        loss = symmetric_kl(teacher, student)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([self.embeddings, *self.lora], self.cfg.max_grad_norm)
        self.optim.step()
        self.sched.step()
        self.optim.zero_grad(set_to_none=True)
        return loss.item(), grad_norm.item(), 1 - self.press.compression_ratio

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.set_adapter(self.adapter)
        self.model.save_pretrained(path)  # adapter-only save with transformers' PEFT integration
        save_file({"restore_embeddings": self.embeddings.detach().to(torch.bfloat16).cpu()}, os.path.join(path, EMBEDDINGS_FILE))
        with open(os.path.join(path, "train_config.json"), "w") as f:
            json.dump(asdict(self.cfg), f, indent=2)

    def train(self, on_save=None):
        order, t0, window = [], time.time(), []
        for step in range(self.cfg.steps):
            if not order:
                order = list(range(len(self.data)))
                self.rng.shuffle(order)
            loss, gnorm, budget = self.step(self.data[order.pop()])
            window.append(loss)
            if (step + 1) % self.cfg.log_every == 0:
                rate = (step + 1) / (time.time() - t0)
                print(
                    f"step {step + 1}/{self.cfg.steps} loss {sum(window) / len(window):.4f} gnorm {gnorm:.3f} "
                    f"budget {budget:.3f} lr {self.sched.get_last_lr()[0]:.2e} {rate:.2f} it/s",
                    flush=True,
                )
                window = []
            if (step + 1) % self.cfg.save_every == 0 or step + 1 == self.cfg.steps:
                path = os.path.join(self.cfg.output_dir, f"step{step + 1}" if step + 1 < self.cfg.steps else "final")
                self.save(path)
                if on_save:
                    on_save()
