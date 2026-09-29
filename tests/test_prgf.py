"""CPU tests for PRGF masking and the press, on a tiny random Qwen3 model."""

import copy

import pytest
import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file
from transformers import AutoTokenizer, DynamicCache, Qwen3Config, Qwen3ForCausalLM, pipeline

from prgf.masking import attention_bias, region_ids, restore_allowed, restore_pass_provider
from prgf.press import EMBEDDINGS_FILE, PartitionedRestoreKVPress

TOKENIZER = "Qwen/Qwen3-0.6B"
N_RESTORE = 8


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
        attn_implementation="sdpa",
    )
    m = Qwen3ForCausalLM(config).eval()
    m.config.name_or_path = TOKENIZER  # KVzip loads the tokenizer from here
    return m


@pytest.fixture(scope="module")
def adapter_dir(model, tmp_path_factory):
    path = tmp_path_factory.mktemp("adapter")
    lora = LoraConfig(
        r=4, lora_alpha=8, init_lora_weights=False,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    get_peft_model(copy.deepcopy(model), lora).save_pretrained(path)
    save_file({"restore_embeddings": torch.randn(N_RESTORE, model.config.hidden_size)}, str(path / EMBEDDINGS_FILE))
    return str(path)


def test_regions_are_contiguous_and_balanced():
    ids = region_ids(100, 7)
    assert ids[0] == 0 and ids[-1] == 6 and (ids.diff() >= 0).all()
    sizes = torch.bincount(ids)
    assert sizes.max() - sizes.min() <= 1


def test_restore_mask_matches_spec():
    torch.manual_seed(0)
    h, t, n = 2, 23, N_RESTORE
    kept = torch.rand(h, t) < 0.3
    allowed = restore_allowed(kept, n, "prgf")
    regions = region_ids(t, n - 1)
    assert allowed.shape == (h, n, t + n)
    for j in range(n - 1):  # local slots
        assert torch.equal(allowed[:, j, :t], kept | (regions == j))
        assert torch.equal(allowed[:, j, t:], torch.nn.functional.one_hot(torch.tensor(j), n).bool().expand(h, n))
    assert allowed[:, n - 1].all()  # global slot: full context + all locals + itself
    # every evicted position is directly readable by exactly one local slot
    evicted = ~kept
    readers = (allowed[:, : n - 1, :t] & evicted[:, None]).sum(1)
    assert torch.equal(readers, evicted.long())


def _prefill(model, length=40):
    torch.manual_seed(1)
    ids = torch.randint(0, 1000, (1, length))
    cache = DynamicCache()
    with torch.no_grad():
        model.model(input_ids=ids, past_key_values=cache)
    return cache


def _restore_pass(model, cache, embeddings, provider=None):
    t = cache.get_seq_length()
    pos = torch.arange(t, t + embeddings.shape[0])
    cache = copy.deepcopy(cache)
    kwargs = dict(inputs_embeds=embeddings[None], past_key_values=cache, position_ids=pos[None], cache_position=pos)
    with torch.no_grad():
        if provider is None:
            out = model.model(**kwargs).last_hidden_state
        else:
            with attention_bias(provider):
                out = model.model(**kwargs).last_hidden_state
    return out[0], cache


def test_causal_mode_matches_default_attention(model):
    cache = _prefill(model)
    emb = torch.randn(N_RESTORE, model.config.hidden_size)
    kept = torch.rand(model.config.num_hidden_layers, 2, cache.get_seq_length()) < 0.5
    ref, _ = _restore_pass(model, cache, emb)
    out, _ = _restore_pass(model, cache, emb, restore_pass_provider(kept, N_RESTORE, 2, "causal"))
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_local_slot_only_sees_own_evicted_region(model):
    cache = _prefill(model, length=42)
    t, n = cache.get_seq_length(), N_RESTORE
    emb = torch.randn(n, model.config.hidden_size)
    kept = torch.zeros(model.config.num_hidden_layers, 2, t, dtype=torch.bool)
    kept[..., :4] = True  # sinks
    provider = restore_pass_provider(kept, n, 2, "prgf")
    base, _ = _restore_pass(model, cache, emb, provider)

    target = int((region_ids(t, n - 1) == 3).nonzero()[0])  # an evicted position of region 3
    perturbed = copy.deepcopy(cache)
    for layer in perturbed.layers:
        layer.keys[:, :, target] += 3.0
        layer.values[:, :, target] += 3.0
    out, _ = _restore_pass(model, perturbed, emb, provider)

    changed = (out - base).abs().amax(-1) > 1e-6
    assert changed.tolist() == [j in (3, n - 1) for j in range(n)]  # only R_3 and G read it


@pytest.mark.parametrize("mode", ["prgf", "causal"])
def test_press_end_to_end_budget(model, adapter_dir, mode):
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    pipe = pipeline("kv-press-text-generation", model=model, tokenizer=tok, device="cpu")
    press = PartitionedRestoreKVPress(compression_ratio=0.75, adapter=adapter_dir, mask_mode=mode)
    context = " ".join(f"word{i}" for i in range(150))
    cache = DynamicCache()
    answer = pipe(context, question="Which word?", press=press, cache=cache, max_new_tokens=3)["answer"]
    assert isinstance(answer, str)

    cfg = model.config
    n_layers, n_heads = cfg.num_hidden_layers, cfg.num_key_value_heads
    ctx_len = cache.get_seq_length() - N_RESTORE
    masked = sum(len(layer.self_attn.masked_key_indices[2]) for layer in model.model.layers)
    kept = n_layers * n_heads * ctx_len - masked
    budget = n_layers * n_heads * ctx_len * (1 - 0.75)
    # kept context + restore slots == budget (up to integer rounding)
    assert abs(kept + N_RESTORE * n_layers * n_heads - budget) <= 1


def test_cached_tokenizer_is_reused():
    import kvpress.presses.kvzip_press as kz

    from prgf import speedups

    speedups.enable()
    assert kz.AutoTokenizer.from_pretrained(TOKENIZER) is kz.AutoTokenizer.from_pretrained(TOKENIZER)


def test_restore_mask_batched_over_layers_matches_per_layer():
    kept = torch.rand(3, 2, 29) < 0.4
    for mode in ("prgf", "causal"):
        batched = restore_allowed(kept, N_RESTORE, mode)
        assert torch.equal(batched, torch.stack([restore_allowed(k, N_RESTORE, mode) for k in kept]))


def test_exchange_mask_opens_only_local_to_global_in_late_layers():
    kept = torch.rand(6, 2, 31) < 0.3
    n, t = N_RESTORE, 31
    v1 = restore_pass_provider(kept, n, 1, "prgf")
    v2 = restore_pass_provider(kept, n, 1, "prgf", exchange_from_layer=4)
    for layer in range(6):
        a, b = v1(layer, n, t + n), v2(layer, n, t + n)
        if layer < 4:
            assert torch.equal(a, b)
        else:
            diff = (a != b).nonzero().tolist()  # (batch, head, query, key)
            assert {(q, k) for _, _, q, k in diff} == {(j, t + n - 1) for j in range(n - 1)}
            assert b[..., : n - 1, t + n - 1].all() and not a[..., : n - 1, t + n - 1].any()


def test_exchange_lets_locals_see_other_regions_through_global(model):
    cache = _prefill(model, length=42)
    t, n = cache.get_seq_length(), N_RESTORE
    emb = torch.randn(n, model.config.hidden_size)
    kept = torch.zeros(model.config.num_hidden_layers, 2, t, dtype=torch.bool)
    kept[..., :4] = True
    target = int((region_ids(t, n - 1) == 3).nonzero()[0])
    perturbed = copy.deepcopy(cache)
    for layer in perturbed.layers:
        layer.keys[:, :, target] += 3.0
        layer.values[:, :, target] += 3.0
    provider = restore_pass_provider(kept, n, 2, "prgf", exchange_from_layer=1)
    base, _ = _restore_pass(model, cache, emb, provider)
    out, _ = _restore_pass(model, perturbed, emb, provider)
    assert ((out - base).abs().amax(-1) > 1e-6).all()  # every local slot now receives region-3 info via G
