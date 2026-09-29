"""Self-distillation data, following the RestoreKV recipe (paper, Table E).

Contexts come from LongAlpaca papers, PG-19 book chunks and Tulu-3 FLAN prompts. The target model
itself (the full-cache teacher) writes five questions per long context (factual, summarization,
multi-hop, method, comparison) and then answers each question with the full context, greedily,
with at most 512 tokens. FLAN prompts carry their own instruction, so they get one empty question.
No RULER-style synthetic data is used: evaluation tasks stay out of training.

Prompts are tokenized with kvpress' own pipeline preprocessing, so training sees exactly the token
layout used at evaluation time: [chat prefix + context] [question + suffix] [answer].
"""

import json
import random
import re

QUESTION_PROMPT = (
    "\n\n---\nWrite five diverse questions that can be answered using the text above: "
    "(1) a factual detail question, (2) a summarization question, (3) a multi-hop question that combines "
    "information from different parts of the text, (4) a question about a method, process or sequence of events, "
    "(5) a comparison question. Output only a JSON list of five strings."
)


def kvpress_inputs(tokenizer, context: str, questions: list[str]) -> tuple[list[int], list[list[int]]]:
    """Token ids exactly as kvpress' KVPressTextGenerationPipeline.preprocess builds them (no answer prefix).

    Re-implemented (and tested against kvpress) so the vLLM data image does not need kvpress.
    """
    if tokenizer.chat_template is None:
        context_text, suffix = (tokenizer.bos_token or "") + context, "\n"
    else:
        separator = "#" * (len(context) + 10)
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": context + separator}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        context_text, suffix = text.split(separator)
    context_ids = tokenizer.encode(context_text, add_special_tokens=False)
    return context_ids, [tokenizer.encode(q + suffix, add_special_tokens=False) for q in questions]


def _longalpaca(n, tokenizer, min_tokens, max_tokens, rng):
    from datasets import load_dataset

    rows = load_dataset("Yukang/LongAlpaca-12k", split="train")
    order = list(range(len(rows)))
    rng.shuffle(order)
    out = []
    for i in order:
        text = rows[i]["instruction"]
        m = re.search(r"The paper begins\.(.*)Now the paper ends\.", text, re.S)
        if m is None:
            continue
        body = m.group(1).strip()
        if min_tokens <= len(tokenizer.encode(body)) <= max_tokens:
            out.append({"source": "longalpaca", "context": body})
        if len(out) == n:
            break
    return out


def _pg19(n, tokenizer, chunk_tokens, rng, chunks_per_book=2):
    from datasets import load_dataset

    books = load_dataset("emozilla/pg19", split="train", streaming=True).shuffle(seed=rng.randint(0, 10**6))
    out = []
    for book in books:
        ids = tokenizer.encode(book["text"])
        if len(ids) < 4 * chunk_tokens:
            continue
        starts = rng.sample(range(chunk_tokens, len(ids) - chunk_tokens), chunks_per_book)  # skip front matter
        for s in starts:
            length = rng.randint(chunk_tokens // 2, chunk_tokens)
            out.append({"source": "pg19", "context": tokenizer.decode(ids[s : s + length])})
        if len(out) >= n:
            return out[:n]
    return out


def _flan(n, tokenizer, min_tokens, max_tokens, rng):
    from datasets import load_dataset

    rows = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
    rows = rows.filter(lambda r: "flan" in r["source"]).shuffle(seed=rng.randint(0, 10**6), buffer_size=10_000)
    out = []
    for row in rows:
        prompt = row["messages"][0]["content"]
        if min_tokens <= len(tokenizer.encode(prompt)) <= max_tokens:
            out.append({"source": "flan", "context": prompt, "questions": [""]})
        if len(out) == n:
            break
    return out


def collect_contexts(tokenizer, n_longalpaca, n_pg19, n_flan, max_tokens=8192, seed=0):
    rng = random.Random(seed)
    data = (
        _longalpaca(n_longalpaca, tokenizer, 1024, max_tokens, rng)
        + _pg19(n_pg19, tokenizer, 4096, rng)
        + _flan(n_flan, tokenizer, 256, max_tokens, rng)
    )
    rng.shuffle(data)
    for i, d in enumerate(data):
        d["id"] = i
    return data


def parse_questions(text: str, k: int = 5) -> list[str]:
    try:
        m = re.search(r"\[.*\]", text, re.S)
        qs = [str(q).strip() for q in json.loads(m.group(0))] if m else []
    except (json.JSONDecodeError, TypeError):
        qs = []
    if not qs:  # fallback: numbered / bulleted lines
        qs = [re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", line).strip() for line in text.splitlines()]
        qs = [q for q in qs if q.endswith("?")]
    return [q for q in qs if q][:k]


def generate_qa(llm, tokenizer, data: list[dict], max_answer_tokens=512) -> list[dict]:
    """Fill in teacher questions (when missing) and teacher answers with a vLLM engine."""
    from vllm import SamplingParams

    greedy = SamplingParams(temperature=0.0, max_tokens=max_answer_tokens)
    need_q = [d for d in data if "questions" not in d]
    prompts = [kvpress_inputs(tokenizer, d["context"] + QUESTION_PROMPT, [""]) for d in need_q]
    outs = llm.generate([{"prompt_token_ids": c + q[0]} for c, q in prompts], greedy)
    for d, o in zip(need_q, outs):
        d["questions"] = parse_questions(o.outputs[0].text)
    data = [d for d in data if d["questions"]]

    requests, owners = [], []
    for d in data:
        ctx, qs = kvpress_inputs(tokenizer, d["context"], d["questions"])
        for qi, q in enumerate(qs):
            requests.append({"prompt_token_ids": ctx + q})
            owners.append((d, qi))
        d["answer_ids"] = [None] * len(qs)
    for (d, qi), o in zip(owners, llm.generate(requests, greedy)):
        d["answer_ids"][qi] = list(o.outputs[0].token_ids)  # includes <|im_end|> when the answer finished
    return data
