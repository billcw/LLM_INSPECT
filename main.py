"""
main.py - FastAPI backend for the next-token demo.

Tokenization, prediction (full logit vector for client-side temperature),
generation with tracing, dual-lens layer analysis, and attention patterns.

Requires jacobian_cache.pt from fit_jacobian.py for the J-lens.
CPU only, fp32. Windows 11, Python 3.14.
"""

from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from transformers import AutoModelForCausalLM, AutoTokenizer

CHECKPOINT = "HuggingFaceTB/SmolLM2-135M"
CACHE_PATH = Path("jacobian_cache.pt")
NAMED_TOKENS = 200      # token strings sent to the client up front
ATTN_MAX_TOKENS = 32    # attention payload grows with the square of this

# Serving only - no backward pass anywhere. Set globally here, and ALSO
# enforced per-function with @torch.no_grad() below, because this global
# was observed not to hold across a --reload re-import.
torch.set_grad_enabled(False)

print(f"Loading {CHECKPOINT} ...")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

# attn_implementation="eager" is REQUIRED for output_attentions=True.
# The default 'sdpa' backend fuses the attention computation and never
# materializes the weight matrix, so it returns an empty tuple rather
# than raising. Eager is slightly slower but this is a teaching tool.
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT,
    dtype=torch.float32,
    attn_implementation="eager",
)
model.eval()
print(f"Model loaded. Attention implementation: "
      f"{model.config._attn_implementation}")

N_LAYERS = model.config.num_hidden_layers
HIDDEN = model.config.hidden_size
VOCAB = model.config.vocab_size
N_HEADS = model.config.num_attention_heads
N_KV_HEADS = model.config.num_key_value_heads

JACOBIANS = None
JACOBIAN_META = None
if CACHE_PATH.exists():
    blob = torch.load(CACHE_PATH, weights_only=False)
    if blob.get("checkpoint") == CHECKPOINT and blob.get("n_layers") == N_LAYERS:
        JACOBIANS = blob["jacobians"]
        JACOBIAN_META = {
            "corpus_size": blob.get("corpus_size"),
            "convention": blob.get("convention"),
        }
        print(f"Jacobian cache loaded ({blob.get('corpus_size')} prompts).")
    else:
        print("Jacobian cache does not match this model - ignoring.")
else:
    print("No jacobian_cache.pt found - J-lens disabled. Run fit_jacobian.py.")

app = FastAPI()


def token_list(token_ids):
    return [
        {"position": i, "id": int(tid), "text": tokenizer.decode([tid])}
        for i, tid in enumerate(token_ids)
    ]


@app.get("/api/tokenize")
def tokenize(prompt: str = Query(..., min_length=0)):
    """
    Tokenization only - the model is never invoked.

    Exists so the UI can show live token chips while typing without paying
    for a 30-layer forward pass on every keystroke.
    """
    if not prompt:
        return {"prompt": prompt, "tokens": []}
    return {"prompt": prompt, "tokens": token_list(tokenizer.encode(prompt))}


@app.get("/api/decode")
def decode(id: int = Query(..., ge=0)):
    """Single token id -> its text. Used for rank lookups and tail hits."""
    if id >= VOCAB:
        return {"id": id, "text": None, "error": "id out of range"}
    return {"id": id, "text": tokenizer.decode([id])}


@app.get("/api/predict")
@torch.no_grad()
def predict(prompt: str = Query(..., min_length=1)):
    """
    One forward pass. Returns the COMPLETE logit vector.

    No temperature parameter: temperature does not affect logits, and the
    client can apply it exactly given the full vector. Called once per
    prompt, not once per slider move.

    Ranking is by logit. Because dividing by a positive temperature is a
    monotonic transformation, that ranking is identical at every T > 0.
    """
    token_ids = tokenizer.encode(prompt)
    outputs = model(torch.tensor([token_ids]))
    logits = outputs.logits[0, -1, :]

    order = torch.argsort(logits, descending=True)
    named = [
        {"id": int(tid), "text": tokenizer.decode([tid])}
        for tid in order[:NAMED_TOKENS]
    ]

    return {
        "prompt": prompt,
        "vocab_size": VOCAB,
        "n_layers": N_LAYERS,
        "hidden_size": HIDDEN,
        "tokens": token_list(token_ids),
        "named_tokens": named,
        # Rounded to 3 decimals: a 0.001 logit difference is far below
        # display precision, and it roughly halves the payload size.
        "logits": [round(float(x), 3) for x in logits],
    }


@app.get("/api/attention")
@torch.no_grad()
def attention(prompt: str = Query(..., min_length=1)):
    """
    Attention weights for every layer and head.

    outputs.attentions is a tuple of N_LAYERS tensors, each of shape
    (batch, n_heads, seq, seq). Entry [h, i, j] is how much of query
    position i's attention went to key position j, in head h.

    Rows sum to 1. The upper triangle is exactly zero because of causal
    masking - position i cannot attend to anything after itself.

    Note on GQA: weights are per QUERY head (9 of them). The 3 key/value
    heads are shared across query heads underneath.
    """
    token_ids = tokenizer.encode(prompt)
    truncated = len(token_ids) > ATTN_MAX_TOKENS
    if truncated:
        token_ids = token_ids[:ATTN_MAX_TOKENS]

    ids = torch.tensor([token_ids])
    outputs = model(ids, output_attentions=True)

    base = {
        "prompt": prompt,
        "tokens": token_list(token_ids),
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "n_kv_heads": N_KV_HEADS,
        "truncated": truncated,
        "max_tokens": ATTN_MAX_TOKENS,
        "attn_implementation": model.config._attn_implementation,
    }

    # Guard: sdpa returns an empty tuple here rather than raising, which
    # previously produced a confusing undefined-length error in the UI.
    if not outputs.attentions:
        return {
            **base,
            "attentions": [],
            "error": (
                "No attention weights returned. The model must be loaded "
                "with attn_implementation='eager'. Current implementation: "
                f"{model.config._attn_implementation}."
            ),
        }

    layers = []
    for layer_attn in outputs.attentions:
        heads = layer_attn[0]                     # (n_heads, seq, seq)
        layers.append([
            [[round(float(v), 4) for v in row] for row in head]
            for head in heads
        ])

    return {**base, "attentions": layers}


@torch.no_grad()
def generate_once(prompt: str, n_tokens: int, temperature: float,
                  seed: Optional[int], trace: bool = False) -> dict:
    """
    Generate n_tokens autoregressively.

    Deliberately recomputes the full sequence each step (no KV cache).
    Slower, but the computation stays obvious.
    """
    if seed is not None:
        torch.manual_seed(seed)

    ids = torch.tensor([tokenizer.encode(prompt)])
    generated_ids = []
    steps = []

    for step_index in range(n_tokens):
        logits = model(ids).logits[0, -1, :]
        raw_probs = torch.softmax(logits, dim=-1)

        if temperature <= 0.0:
            next_id = int(torch.argmax(logits))
        else:
            sample_probs = torch.softmax(logits / temperature, dim=-1)
            next_id = int(torch.multinomial(sample_probs, num_samples=1))

        if trace:
            order = torch.argsort(logits, descending=True)
            rank = int((order == next_id).nonzero()[0, 0]) + 1
            top = torch.topk(logits, 5)
            steps.append({
                "step": step_index + 1,
                "context": tokenizer.decode(ids[0].tolist()),
                "context_length": int(ids.shape[1]),
                "candidates": [
                    {"token": tokenizer.decode([tid]),
                     "probability": float(raw_probs[tid])}
                    for tid in top.indices
                ],
                "chosen": {
                    "token": tokenizer.decode([next_id]),
                    "probability": float(raw_probs[next_id]),
                    "rank": rank,
                },
            })

        generated_ids.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)

    return {
        "generated_text": tokenizer.decode(generated_ids),
        "generated_tokens": [tokenizer.decode([t]) for t in generated_ids],
        "steps": steps,
    }


@app.get("/api/generate")
def generate(
    prompt: str = Query(..., min_length=1),
    n_tokens: int = Query(10, ge=1, le=50),
    temperature: float = Query(0.0, ge=0.0, le=5.0),
    runs: int = Query(1, ge=1, le=10),
    seed: Optional[int] = Query(None),
):
    """
    Run generation `runs` times with identical settings.

    At temperature 0, every run is identical - the model is deterministic.
    Above 0, runs diverge because a sample is drawn from the distribution.
    With a fixed seed, even the sampled runs become reproducible.

    Only the first run is traced, to keep the response small.

    No @torch.no_grad() here - generate_once carries it, and that is
    where the model is actually invoked.
    """
    results = [
        generate_once(prompt, n_tokens, temperature, seed,
                      trace=(run_index == 0))
        for run_index in range(runs)
    ]
    unique_outputs = len({r["generated_text"] for r in results})

    return {
        "prompt": prompt,
        "n_tokens": n_tokens,
        "temperature": temperature,
        "runs": runs,
        "seed": seed,
        "unique_outputs": unique_outputs,
        "all_identical": unique_outputs == 1,
        "forward_passes": runs * n_tokens,
        "results": results,
    }


def activations_for(ids: torch.Tensor):
    """
    Forward pass, additionally capturing the PRE-NORM output of the last
    decoder layer.

    hidden_states[-1] has already had the final RMSNorm applied, so it
    cannot serve as the stage-30 source for the J-lens - the Jacobians
    were fitted against the pre-norm tensor.
    """
    store = {}

    def hook(module, args, output):
        store["h"] = output[0] if isinstance(output, tuple) else output

    handle = model.model.layers[-1].register_forward_hook(hook)
    try:
        outputs = model(ids, output_hidden_states=True)
    finally:
        handle.remove()
    return outputs, store["h"]


@app.get("/api/lenses")
@torch.no_grad()
def lenses(
    prompt: str = Query(..., min_length=1),
    k: int = Query(3, ge=1, le=8),
):
    """
    Read out predictions from every layer using two different lenses.

    LOGIT LENS
        readout(h_L) = lm_head(norm(h_L))
        Pretends layers L+1..30 do not exist.

    JACOBIAN LENS
        readout(h_L) = lm_head(norm(J_L @ h_L))
        Approximates the remaining layers with one averaged linear map.

    At stage 30 the Jacobian is the identity, so the J-lens must
    reproduce the model's real logits exactly. That is the diagnostic.
    """
    token_ids = tokenizer.encode(prompt)
    ids = torch.tensor([token_ids])

    outputs, prenorm_final = activations_for(ids)
    hidden_states = outputs.hidden_states
    true_logits = outputs.logits[0, -1, :]

    target_id = int(torch.argmax(true_logits))
    target_token = tokenizer.decode([target_id])
    final_norm = model.model.norm

    def describe(logits: torch.Tensor):
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(logits, k)
        order = torch.argsort(logits, descending=True)
        return {
            "top": [
                {"token": tokenizer.decode([tid]),
                 "probability": float(probs[tid])}
                for tid in top.indices
            ],
            "target_rank": int((order == target_id).nonzero()[0, 0]) + 1,
            "target_probability": float(probs[target_id]),
        }

    stages = []
    jacobian_final_error = None

    for stage in range(N_LAYERS + 1):
        vec = hidden_states[stage][0, -1, :]

        # hidden_states[N_LAYERS] is already normed; do not norm twice.
        if stage == N_LAYERS:
            logit_lens_logits = model.lm_head(vec)
        else:
            logit_lens_logits = model.lm_head(final_norm(vec))

        entry = {
            "index": stage,
            "label": "embeddings" if stage == 0 else f"layer {stage}",
            "vector_norm": float(vec.norm()),
            "logit_lens": describe(logit_lens_logits),
            "jacobian_lens": None,
        }

        # Jacobians were fitted for stages 1..30 only, not the embeddings.
        if JACOBIANS is not None and stage >= 1:
            source = prenorm_final[0, -1, :] if stage == N_LAYERS else vec
            j_logits = model.lm_head(final_norm(JACOBIANS[stage] @ source))
            entry["jacobian_lens"] = describe(j_logits)
            if stage == N_LAYERS:
                jacobian_final_error = float(
                    (j_logits - true_logits).abs().max())

        stages.append(entry)

    def first_in_top5(key):
        return next(
            (s["index"] for s in stages
             if s[key] is not None and s[key]["target_rank"] <= 5),
            None,
        )

    return {
        "prompt": prompt,
        "tokens": token_list(token_ids),
        "n_stages": N_LAYERS + 1,
        "target_token": target_token,
        "target_probability": float(
            torch.softmax(true_logits, dim=-1)[target_id]),
        "jacobian_available": JACOBIANS is not None,
        "jacobian_meta": JACOBIAN_META,
        "diagnostics": {"jacobian_final_error": jacobian_final_error},
        "first_top5_logit_lens": first_in_top5("logit_lens"),
        "first_top5_jacobian_lens": first_in_top5("jacobian_lens"),
        "stages": stages,
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": CHECKPOINT,
        "attn_implementation": model.config._attn_implementation,
        "jacobian_lens": JACOBIANS is not None,
    }


# Must come AFTER the API routes above - this mount catches all other paths.
app.mount("/", StaticFiles(directory="static", html=True), name="static")