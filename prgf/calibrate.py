"""Reference-query second moment G = E[q q^T / d] per (layer, KV head) for the transport cost.

Queries are post-RoPE queries of question/answer tokens (the tokens that will read the restore memory),
collected with the full context on training samples; the query heads sharing a KV head are pooled (GQA).
No evaluation data and no inference-time question is involved.
"""

import torch
from kvpress.utils import get_query_states

from prgf.data import kvpress_inputs


@torch.no_grad()
def query_second_moment(model, tokenizer, samples: list[dict], max_answer_tokens: int = 256) -> torch.Tensor:
    cfg = model.config
    L, H, d = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim
    acc = torch.zeros(L, H, d, d, device=model.device)
    count = torch.zeros(L, device=model.device)
    state = {"start": 0}

    def hook(module, args, kwargs, output):
        cos, sin = kwargs["position_embeddings"]
        s = state["start"]
        q = get_query_states(module, kwargs["hidden_states"][:, s:], (cos[:, s:], sin[:, s:])).float()
        q = q.view(q.shape[0], H, -1, q.shape[-2], d).transpose(1, 0).reshape(H, -1, d)  # pool GQA group + tokens
        acc[module.layer_idx] += q.transpose(1, 2) @ q / d
        count[module.layer_idx] += q.shape[1]

    hooks = [layer.self_attn.register_forward_hook(hook, with_kwargs=True) for layer in model.model.layers]
    try:
        for sample in samples:
            ctx, q_ids = kvpress_inputs(tokenizer, sample["context"], sample["questions"])
            ids = ctx + q_ids[0] + sample["answer_ids"][0][:max_answer_tokens]
            state["start"] = len(ctx)
            model.model(input_ids=torch.tensor([ids], device=model.device))
    finally:
        for h in hooks:
            h.remove()
    return (acc / count[:, None, None, None]).cpu()
