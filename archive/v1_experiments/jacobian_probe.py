"""
jacobian_probe.py - timing probe for Jacobian lens feasibility.

Measures how long it takes to compute rows of  d h_final / d h_L  for a
single layer and a single prompt, then extrapolates to the full job.

Does NOT compute a full Jacobian. Does NOT touch main.py.
Windows 11, Python 3.14, CPU only, fp32.
"""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CHECKPOINT = "HuggingFaceTB/SmolLM2-135M"
PROMPT = "The capital of Japan is called"
PROBE_LAYER = 15          # residual stream stage to differentiate from
SAMPLE_ROWS = 16          # rows to time before extrapolating
BATCHED_ROWS = 32         # rows to attempt in the batched path

print(f"Loading {CHECKPOINT} ...")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, dtype=torch.float32)
model.eval()

# We differentiate with respect to an injected activation, never the
# weights. Turning off parameter gradients shrinks the autograd graph.
model.requires_grad_(False)

n_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size
print(f"Model loaded. {n_layers} layers, hidden size {hidden_size}.\n")

ids = torch.tensor([tokenizer.encode(PROMPT)])
n_tokens = ids.shape[1]
print(f"Prompt: {PROMPT!r}  ({n_tokens} tokens)")
print(f"Differentiating h_final with respect to stage {PROBE_LAYER}\n")


def make_injection_hook(replacement):
    """
    Replace a decoder layer's output hidden state with our own tensor.

    A forward hook that returns a non-None value replaces the module
    output. The layer may return a bare tensor or a tuple depending on
    the transformers version, so handle both.
    """
    def hook(module, args, output):
        if isinstance(output, tuple):
            return (replacement,) + output[1:]
        return replacement
    return hook


# ---------------------------------------------------------------- capture
# hidden_states has n_layers + 1 entries:
#   index 0        = embeddings
#   index i + 1    = output of decoder layer i
# So stage PROBE_LAYER is produced by layers[PROBE_LAYER - 1].
t0 = time.perf_counter()
with torch.no_grad():
    baseline = model(ids, output_hidden_states=True)
forward_time = time.perf_counter() - t0

captured = baseline.hidden_states[PROBE_LAYER].detach().clone()
print(f"Plain forward pass          : {forward_time * 1000:8.1f} ms")
print(f"Captured stage shape        : {tuple(captured.shape)}")

target_module = model.model.layers[PROBE_LAYER - 1]


def run_with_injection():
    """Run the model with our differentiable tensor spliced in."""
    injected = captured.clone().requires_grad_(True)
    handle = target_module.register_forward_hook(make_injection_hook(injected))
    try:
        out = model(ids, output_hidden_states=True)
        h_final = out.hidden_states[-1][0, -1, :]
    finally:
        handle.remove()
    return injected, h_final


# ---------------------------------------------------- sanity: graph exists
injected, h_final = run_with_injection()
print(f"h_final requires grad       : {h_final.requires_grad}")
if not h_final.requires_grad:
    print("\nFAILED: no autograd graph. The hook is not connecting.")
    raise SystemExit(1)

# Confirm the injection reproduces the unmodified model output.
with torch.no_grad():
    reference = baseline.hidden_states[-1][0, -1, :]
injection_error = float((h_final.detach() - reference).abs().max())
print(f"Injection max abs error     : {injection_error:.3e}"
      f"   (should be ~0)\n")


# ------------------------------------------------- path 1: one row at a time
print("-" * 62)
print(f"PATH 1 - sequential backward passes ({SAMPLE_ROWS} rows sampled)")
print("-" * 62)

injected, h_final = run_with_injection()
t0 = time.perf_counter()
for row in range(SAMPLE_ROWS):
    torch.autograd.grad(h_final[row], injected, retain_graph=True)
sequential_time = time.perf_counter() - t0

per_row = sequential_time / SAMPLE_ROWS
one_layer = per_row * hidden_size
print(f"  {SAMPLE_ROWS} rows                    : {sequential_time:8.2f} s")
print(f"  per row                    : {per_row * 1000:8.1f} ms")
print(f"  -> one full Jacobian       : {one_layer:8.1f} s"
      f"   ({one_layer / 60:.1f} min)")
print(f"  -> all {n_layers} layers, 1 prompt  : "
      f"{one_layer * n_layers / 60:8.1f} min")
print(f"  -> all {n_layers} layers, 10 prompts : "
      f"{one_layer * n_layers * 10 / 3600:8.1f} hours")


# ----------------------------------------------------- path 2: batched vmap
print()
print("-" * 62)
print(f"PATH 2 - batched grads via vmap ({BATCHED_ROWS} rows attempted)")
print("-" * 62)

try:
    injected, h_final = run_with_injection()
    basis = torch.zeros(BATCHED_ROWS, hidden_size)
    for row in range(BATCHED_ROWS):
        basis[row, row] = 1.0

    t0 = time.perf_counter()
    batched = torch.autograd.grad(
        outputs=h_final,
        inputs=injected,
        grad_outputs=basis,
        is_grads_batched=True,
        retain_graph=False,
    )[0]
    batched_time = time.perf_counter() - t0

    batched_per_row = batched_time / BATCHED_ROWS
    batched_one_layer = batched_per_row * hidden_size
    speedup = per_row / batched_per_row

    print(f"  result shape               : {tuple(batched.shape)}")
    print(f"  {BATCHED_ROWS} rows                    : {batched_time:8.2f} s")
    print(f"  per row                    : {batched_per_row * 1000:8.1f} ms")
    print(f"  speedup vs sequential      : {speedup:8.1f}x")
    print(f"  -> one full Jacobian       : {batched_one_layer:8.1f} s"
          f"   ({batched_one_layer / 60:.1f} min)")
    print(f"  -> all {n_layers} layers, 1 prompt  : "
          f"{batched_one_layer * n_layers / 60:8.1f} min")
    print(f"  -> all {n_layers} layers, 10 prompts : "
          f"{batched_one_layer * n_layers * 10 / 3600:8.1f} hours")
except Exception as exc:
    print(f"  BATCHED PATH FAILED: {type(exc).__name__}")
    print(f"  {exc}")
    print("  Not fatal - we fall back to path 1.")


print()
print("-" * 62)
print("Extrapolations assume linear scaling in rows, layers, and prompts.")
print("They are estimates from a small sample, not measurements.")
print("-" * 62)