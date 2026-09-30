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


@pytest.mark.parametrize("plus", [False, True])
def test_cached_scores_give_identical_selection(tiny, plus):
    """Second pass over the same sample loads KVzip(+) scores from disk and must select the same KV pairs."""
    root, _ = tiny
    cfg = TrainConfig(
        data_path=str(root / "data.jsonl"), output_dir=str(root / f"out_cache{plus}"), model=str(root / "model"),
        init_adapter=str(root / "adapter"), score_cache_dir=str(root / f"scores{plus}"), steps=1, plus=plus,
    )
    trainer = Trainer(cfg)
    assert trainer.press.kvzip_plus_normalization == plus
    sample = trainer.data[0]
    ctx_ids, _, _, _ = trainer._batch(sample)
    trainer.press.compression_ratio = 0.9
    from transformers import DynamicCache

    with torch.no_grad(), trainer.press(trainer.model):
        trainer.model.model(input_ids=ctx_ids, past_key_values=DynamicCache())
    fresh = trainer.press.kept_mask.clone()
    trainer._save_scores(sample, trainer.press.scores)
    cached = trainer.press.select_from_scores(trainer.model, trainer._load_scores(sample))
    assert torch.equal(fresh, cached)
    assert not cached.all()  # something was actually evicted


def test_resume_matches_uninterrupted_run(tiny):
    """Train 4 steps straight vs 2 steps + resume to 4: identical final weights."""
    from prgf.train import latest_checkpoint

    root, _ = tiny
    common = dict(data_path=str(root / "data.jsonl"), model=str(root / "model"), init_adapter=str(root / "adapter"),
                  steps=4, warmup_steps=1, lr=1e-2, save_every=2, log_every=1)
    straight = Trainer(TrainConfig(output_dir=str(root / "straight"), **common))
    straight.train()

    class Crash(Exception):
        pass

    def crash():  # the job dies right after the step-2 checkpoint
        raise Crash

    with pytest.raises(Crash):
        Trainer(TrainConfig(output_dir=str(root / "resumed"), **common)).train(on_save=crash)
    resume = latest_checkpoint(str(root / "resumed"))
    assert resume.endswith("step2")
    second = Trainer(TrainConfig(output_dir=str(root / "resumed"), resume_from=resume, **common))
    second.train()
    torch.testing.assert_close(second.embeddings, straight.embeddings)
    for a, b in zip(second.lora, straight.lora):
        torch.testing.assert_close(a, b)


def test_reconstruction_pairs_are_region_local_context_spans(tiny):
    from prgf.masking import region_ids

    root, _ = tiny
    cfg = TrainConfig(data_path=str(root / "data.jsonl"), output_dir=str(root / "out_recon"), model=str(root / "model"),
                      init_adapter=str(root / "adapter"), recon_weight=1.0, recon_span=8, steps=2, warmup_steps=1)
    trainer = Trainer(cfg)
    ctx_ids, qa_ids, where, weights = trainer._batch(trainer.data[0])
    ctx = ctx_ids[0].tolist()
    regions = region_ids(len(ctx), trainer.num_restore - 1).tolist()
    pairs = trainer._recon_pairs(ctx)
    assert len(pairs) == trainer.num_restore - 1  # one span per local slot
    for j, (q, a) in enumerate(pairs):
        start = next(i for i in range(len(ctx)) if ctx[i : i + len(a)] == a)
        assert {regions[i] for i in range(start, start + len(a))} == {j}  # span inside region j only
        assert "Repeat the part of the previous context exactly" in trainer.tok.decode(q)
    # QA tokens and reconstruction tokens each contribute a normalized mean (weights sum to 1 + recon_weight)
    torch.testing.assert_close(weights.sum(), torch.tensor(2.0), rtol=1e-5, atol=1e-5)
    trainer.train()  # runs end to end with the extra targets


def test_train_chained_layout_from_v1_checkpoint(tiny):
    root, tok = tiny
    cfg = TrainConfig(data_path=str(root / "data.jsonl"), output_dir=str(root / "out_k2"), model=str(root / "model"),
                      init_adapter=str(root / "adapter"), slots_per_region=2, num_global=2, recon_weight=1.0,
                      recon_span=8, steps=2, warmup_steps=1)
    trainer = Trainer(cfg)
    assert trainer.num_restore == 16 and trainer.press.num_restore_tokens == 16  # budget pays for 16 slots
    trainer.train()
    model = Qwen3ForCausalLM.from_pretrained(root / "model", attn_implementation="sdpa")
    pipe = pipeline("kv-press-text-generation", model=model, tokenizer=tok, device="cpu")
    press = PartitionedRestoreKVPress(compression_ratio=0.8, adapter=f"{cfg.output_dir}/final", slots_per_region=2, num_global=2)
    cache = DynamicCache()
    pipe(" ".join(f"fact{i}" for i in range(150)), question="What?", press=press, cache=cache, max_new_tokens=2)
    assert press.num_restore_tokens == 16


def test_transport_press_and_trainer(tiny):
    root, tok = tiny
    cfg = TrainConfig(data_path=str(root / "data.jsonl"), output_dir=str(root / "out_ot"), model=str(root / "model"),
                      init_adapter=str(root / "adapter"), slots_per_region=2, num_global=2, transport=True,
                      recon_weight=1.0, recon_span=8, steps=2, warmup_steps=1, lr=1e-2)
    trainer = Trainer(cfg)
    assert trainer.press.reserved_per_head == 17  # 16 slots + one pair holding the 16 mass scalars
    before = trainer.embeddings.detach().clone()
    trainer.train()
    assert not torch.equal(before, trainer.embeddings.detach())  # gradient reaches the PRGF init through P

    model = Qwen3ForCausalLM.from_pretrained(root / "model", attn_implementation="sdpa")
    pipe = pipeline("kv-press-text-generation", model=model, tokenizer=tok, device="cpu")
    press = PartitionedRestoreKVPress(compression_ratio=0.75, adapter=f"{cfg.output_dir}/final",
                                      slots_per_region=2, num_global=2, transport=True)
    context = " ".join(f"fact{i}" for i in range(200))
    cache = DynamicCache()
    answer = pipe(context, question="What?", press=press, cache=cache, max_new_tokens=3)["answer"]
    assert isinstance(answer, str)
    T = cache.get_seq_length() - 16
    L, H = model.config.num_hidden_layers, model.config.num_key_value_heads
    masked = sum(len(layer.self_attn.masked_key_indices[2]) for layer in model.model.layers)
    assert abs((L * H * T - masked) + 17 * L * H - L * H * T * 0.25) <= 1  # budget includes the mass scalars
    start, log_mass = model.model.layers[0].self_attn.prgf_slot_bias
    assert start == T and log_mass.shape == (H, 16)
    kept = press.kept_mask[0]
    torch.testing.assert_close(log_mass.exp().sum(-1), (~kept).sum(-1).float(), rtol=1e-4, atol=1e-3)


def test_query_second_moment_is_psd(tiny):
    from prgf.calibrate import query_second_moment

    root, tok = tiny
    model = Qwen3ForCausalLM.from_pretrained(root / "model")
    samples = [json.loads(line) for line in open(root / "data.jsonl")]
    G = query_second_moment(model, tok, samples)
    cfg = model.config
    assert G.shape == (cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, cfg.head_dim)
    torch.testing.assert_close(G, G.transpose(-1, -2))
    assert (torch.linalg.eigvalsh(G) > -1e-5).all()


def test_train_evicted_partition(tiny):
    """Eviction-mass regions: reconstruction spans lie inside the selection-dependent regions; saved adapter
    loads through the press with the same partition."""
    from prgf.masking import partition_ids

    root, tok = tiny
    cfg = TrainConfig(data_path=str(root / "data.jsonl"), output_dir=str(root / "out_evict"), model=str(root / "model"),
                      init_adapter=str(root / "adapter"), slots_per_region=2, num_global=2, partition="evicted",
                      recon_weight=1.0, recon_span=8, steps=2, warmup_steps=1)
    trainer = Trainer(cfg)
    kept = torch.rand(2, 2, 300) < torch.linspace(0.9, 0.1, 300)
    regions = partition_ids(kept, trainer.num_regions, "evicted").tolist()
    ctx = list(range(1000, 1300))
    for j, (_, a) in enumerate(trainer._recon_pairs(ctx, regions)):
        start = ctx.index(a[0])
        assert {regions[i] for i in range(start, start + len(a))} == {j}
    trainer.train()
    model = Qwen3ForCausalLM.from_pretrained(root / "model", attn_implementation="sdpa")
    pipe = pipeline("kv-press-text-generation", model=model, tokenizer=tok, device="cpu")
    press = PartitionedRestoreKVPress(compression_ratio=0.8, adapter=f"{cfg.output_dir}/final", slots_per_region=2,
                                      num_global=2, partition="evicted")
    pipe(" ".join(f"fact{i}" for i in range(150)), question="What?", press=press, cache=DynamicCache(), max_new_tokens=2)
