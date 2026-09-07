"""
Step 1 - environment and model probe.
SmolLM2-135M on CPU, fp32. No GPU, no web server, no UI.
"""

import sys
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

CHECKPOINT = "HuggingFaceTB/SmolLM2-135M"
PROMPT = "The capital of France is"

torch.set_grad_enabled(False)


def print_environment():
    print("=" * 70)
    print("ENVIRONMENT")
    print("=" * 70)
    print(f"python       : {sys.version.split()[0]}")
    print(f"torch        : {torch.__version__}")
    print(f"transformers : {transformers.__version__}")
    print(f"cuda avail   : {torch.cuda.is_available()}  (expected False)")
    print()


def print_config(model):
    c = model.config
    print("=" * 70)
    print(f"CONFIG - {CHECKPOINT}")
    print("=" * 70)
    for field in [
        "model_type", "num_hidden_layers", "hidden_size",
        "intermediate_size", "num_attention_heads", "num_key_value_heads",
        "vocab_size", "max_position_embeddings", "tie_word_embeddings",
        "rms_norm_eps",
    ]:
        print(f"  {field:26s} = {getattr(c, field, '<absent>')}")
    rope = getattr(c, "rope_parameters", None)
    print(f"  {'rope_parameters':26s} = {rope}")

    n_params = sum(p.numel() for p in model.parameters())
    print()
    print(f"  parameter count            = {n_params:,}")
    print(f"  fp32 weight size (MB)      = {n_params * 4 / 1e6:,.1f}")
    print(f"  get_memory_footprint (MB)  = {model.get_memory_footprint() / 1e6:,.1f}")
    print(f"  actual param dtype         = {next(model.parameters()).dtype}")

    tied = torch.equal(model.lm_head.weight, model.model.embed_tokens.weight)
    print(f"  lm_head == embed_tokens    = {tied}")
    print()


def show_tokens(tokenizer, text):
    ids = tokenizer.encode(text)
    print("=" * 70)
    print(f"TOKENIZATION of {text!r}")
    print("=" * 70)
    print(f"  {len(ids)} tokens")
    for pos, tid in enumerate(ids):
        piece = tokenizer.decode([tid])
        print(f"    pos {pos:2d}  id {tid:6d}  {piece!r}")
    print()
    return ids


def top_k_next(model, tokenizer, text, k=10):
    ids = torch.tensor([tokenizer.encode(text)])
    out = model(ids)
    logits = out.logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    top = torch.topk(probs, k)

    print("=" * 70)
    print(f"TOP {k} NEXT TOKENS after {text!r}")
    print("=" * 70)
    print(f"  logits shape {tuple(out.logits.shape)}  (batch, positions, vocab)")
    print()
    for rank, (p, tid) in enumerate(zip(top.values, top.indices), start=1):
        piece = tokenizer.decode([tid])
        bar = "#" * int(p.item() * 50)
        print(f"  {rank:2d}. {p.item():7.4f}  {piece!r:20s} {bar}")
    print()


def causal_mask_test(model, tokenizer):
    base = "The capital of France is"
    appended = "The capital of France is known around the world"
    prepended = "In geography class we learned the capital of France is"

    def hidden_at(text, layer, position):
        ids = torch.tensor([tokenizer.encode(text)])
        out = model(ids, output_hidden_states=True)
        return out.hidden_states[layer][0, position, :]

    base_ids = tokenizer.encode(base)
    france_pos = None
    for pos, tid in enumerate(base_ids):
        if "France" in tokenizer.decode([tid]):
            france_pos = pos
            break

    print("=" * 70)
    print("CAUSAL MASK TEST")
    print("=" * 70)

    if france_pos is None:
        print("  Could not locate 'France' in the token list.")
        print()
        return

    layer = 15
    v_base = hidden_at(base, layer, france_pos)
    v_app = hidden_at(appended, layer, france_pos)

    print(f"  'France' is token position {france_pos}; comparing layer {layer}")
    print(f"  vector dimension = {v_base.shape[0]}")
    print()
    print(f"  APPEND  max abs difference = {(v_base - v_app).abs().max().item():.3e}")
    print(f"          exactly identical  = {torch.equal(v_base, v_app)}")
    print("          (expected: identical)")
    print()

    pre_ids = tokenizer.encode(prepended)
    pre_pos = None
    for pos, tid in enumerate(pre_ids):
        if "France" in tokenizer.decode([tid]):
            pre_pos = pos
            break

    if pre_pos is not None:
        v_pre = hidden_at(prepended, layer, pre_pos)
        print(f"  PREPEND 'France' now at position {pre_pos}")
        print(f"          max abs difference = {(v_base - v_pre).abs().max().item():.3e}")
        print("          (expected: clearly nonzero)")
    print()


def main():
    print_environment()
    print(f"Loading {CHECKPOINT} (cached locally, should be fast)...\n")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, torch_dtype=torch.float32)
    model.eval()

    print_config(model)
    show_tokens(tokenizer, PROMPT)
    top_k_next(model, tokenizer, PROMPT)
    top_k_next(model, tokenizer, "The capital of Japan is")
    causal_mask_test(model, tokenizer)

    print("Done.")


if __name__ == "__main__":
    main()