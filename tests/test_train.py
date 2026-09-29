"""CPU smoke test: a few distillation steps on a tiny model, then reload the adapter in the press."""

import copy
import json

import pytest
import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file
from transformers import AutoTokenizer, DynamicCache, Qwen3Config, Qwen3ForCausalLM, pipeline

from prgf.press import EMBEDDINGS_FILE, PartitionedRestoreKVPress
from prgf.train import TrainConfig, Trainer, symmetric_kl


def test_symmetric_kl_zero_iff_equal():
    x = torch.randn(4, 11)
    assert symmetric_kl(x, x).abs() < 1e-6
    assert symmetric_kl(x, torch.randn(4, 11)) > 0


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny")
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
    )
    model = Qwen3ForCausalLM(config)
    model.save_pretrained(root / "model")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tok.save_pretrained(root / "model")

    lora = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    get_peft_model(copy.deepcopy(model), lora).save_pretrained(root / "adapter")
    save_file({"restore_embeddings": torch.randn(8, 64)}, str(root / "adapter" / EMBEDDINGS_FILE))

    rows = [
        {"id": 0, "context": " ".join(f"fact{i}" for i in range(200)), "questions": ["What?", "Why?"],
         "answer_ids": [[1, 2, 3, 4], [5, 6, 7]]},
        {"id": 1, "context": "short context " * 30, "questions": [""], "answer_ids": [[9, 10, 11]]},
    ]
    with open(root / "data.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    return root, tok


@pytest.mark.parametrize("mode", ["prgf", "causal"])
def test_train_and_reload(tiny, mode):
    root, tok = tiny
    cfg = TrainConfig(
        data_path=str(root / "data.jsonl"), output_dir=str(root / f"out_{mode}"), model=str(root / "model"),
        init_adapter=str(root / "adapter"), mask_mode=mode, steps=3, warmup_steps=1, lr=1e-2, log_every=1,
    )
    trainer = Trainer(cfg)
    before = trainer.embeddings.detach().clone()
    lora_before = [p.detach().clone() for p in trainer.lora]
    trainer.train()
    assert not torch.equal(before, trainer.embeddings.detach())
    assert any(not torch.equal(b, p.detach()) for b, p in zip(lora_before, trainer.lora))

    # the saved adapter loads through the inference press
    model = Qwen3ForCausalLM.from_pretrained(root / "model", attn_implementation="sdpa")
    pipe = pipeline("kv-press-text-generation", model=model, tokenizer=tok, device="cpu")
    press = PartitionedRestoreKVPress(compression_ratio=0.8, adapter=f"{cfg.output_dir}/final", mask_mode=mode)
    cache = DynamicCache()
    pipe(" ".join(f"fact{i}" for i in range(120)), question="What?", press=press, cache=cache, max_new_tokens=2)
    torch.testing.assert_close(press.restore_embeddings.float(), trainer.embeddings.detach().to(torch.bfloat16).float())
    name = press.adapter_name
    loaded = {n: p for n, p in model.named_parameters() if f".{name}." in n}
    trained = {n.replace(f".{trainer.adapter}.", f".{name}."): p for n, p in trainer.model.named_parameters()
               if f".{trainer.adapter}." in n}
    assert loaded.keys() == trained.keys() and loaded
    for n in loaded:
        torch.testing.assert_close(loaded[n].float(), trained[n].float())


def test_kvpress_inputs_matches_pipeline():
    from types import SimpleNamespace

    from kvpress.pipeline import KVPressTextGenerationPipeline

    from prgf.data import kvpress_inputs

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    context, questions = "Some long context.\nWith lines ### and symbols.", ["Q1?", ""]
    ref = KVPressTextGenerationPipeline.preprocess(SimpleNamespace(tokenizer=tok), context, questions, "", 10**9)
    ctx, qs = kvpress_inputs(tok, context, questions)
    assert ctx == ref["context_ids"][0].tolist()
    assert qs == [q[0].tolist() for q in ref["questions_ids"]]


def test_cached_scores_give_identical_selection(tiny):
    """Second pass over the same sample loads KVzip scores from disk and must select the same KV pairs."""
    root, _ = tiny
    cfg = TrainConfig(
        data_path=str(root / "data.jsonl"), output_dir=str(root / "out_cache"), model=str(root / "model"),
        init_adapter=str(root / "adapter"), score_cache_dir=str(root / "scores"), steps=1,
    )
    trainer = Trainer(cfg)
    sample = trainer.data[0]
    ctx_ids, _, _ = trainer._batch(sample)
    trainer.press.compression_ratio = 0.9
    from transformers import DynamicCache

    with torch.no_grad(), trainer.press(trainer.model):
        trainer.model.model(input_ids=ctx_ids, past_key_values=DynamicCache())
    fresh = trainer.press.kept_mask.clone()
    trainer._save_scores(sample, trainer.press.scores)
    cached = trainer.press.select_from_scores(trainer.model, trainer._load_scores(sample))
    assert torch.equal(fresh, cached)
    assert not cached.all()  # something was actually evicted
