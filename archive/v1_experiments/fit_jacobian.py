"""
fit_jacobian.py - fit and cache Jacobian matrices for the J-lens.

For each stage L, computes the average of
    d h_final[last] / d h_L[last]
across a corpus, where h_final is the PRE-NORM output of the last
decoder layer.

Phase 1 verifies the machinery in seconds and aborts on failure.
Phase 2 does the expensive fit only if phase 1 passes.

Windows 11, Python 3.14, CPU only, fp32.
"""

import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CHECKPOINT = "HuggingFaceTB/SmolLM2-135M"
CACHE_PATH = Path("jacobian_cache.pt")
CHUNK_ROWS = 64

CORPUS = [
    "The capital of Japan is called",
    "She opened the door and found",
    "The three main causes of the war were",
    "To install the package, first run",
    "In 1969, astronauts landed on",
    "He asked her whether she had",
    "Water boils at a temperature of",
    "The company reported quarterly earnings of",
    "My favorite thing about winter is",
    "The function returns a value of type",
    "Scientists have long believed that the",
    "Before you begin, make sure you have",
]

print(f"Loading {CHECKPOINT} ...")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, dtype=torch.float32)
model.eval()
model.requires_grad_(False)

n_layers = model.config.num_hidden_layers
hidden = model.config.hidden_size
final_layer = model.model.layers[n_layers - 1]

print(f"Model loaded. {n_layers} layers, hidden size {hidden}.\n")


def unwrap(output):
    """Decoder layers return a bare tensor or a tuple, depending on version."""
    return output[0] if isinstance(output, tuple) else output


def rewrap(output, replacement):
    return (replacement,) + output[1:] if isinstance(output, tuple) else replacement


def capture_hook(store):
    def hook(module, args, output):
        store["h"] = unwrap(output)
    return hook


def injection_hook(replacement):
    def hook(module, args, output):
        return rewrap(output, replacement)
    return hook


def baseline_run(ids):
    """
    Normal forward pass, additionally capturing the PRE-NORM output of
    the last decoder layer - which hidden_states does not expose.
    """
    store = {}
    handle = final_layer.register_forward_hook(capture_hook(store))
    try:
        with torch.no_grad():
            outputs = model(ids, output_hidden_states=True)
    finally:
        handle.remove()
    return outputs, store["h"]


def source_activation(outputs, prenorm_final, stage):
    """
    The activation to inject at `stage`.

    hidden_states[i] is the INPUT to layer i, i.e. the output of layer
    i-1, for i = 1..29. hidden_states[30] is norm(layer 29 output), so
    for stage 30 we use the captured pre-norm tensor instead.
    """
    if stage == n_layers:
        return prenorm_final.detach().clone()
    return outputs.hidden_states[stage].detach().clone()


def injected_run(ids, stage, injected):
    """
    Forward pass with `injected` spliced in as layer (stage-1)'s output.
    Returns the pre-norm final hidden state at the last position.

    For stage == n_layers both hooks land on the same module. The
    injection hook is registered first, so the capture hook sees the
    replaced output - which is what makes the stage-30 Jacobian the
    identity by construction.
    """
    store = {}
    handles = [
        model.model.layers[stage - 1].register_forward_hook(injection_hook(injected)),
        final_layer.register_forward_hook(capture_hook(store)),
    ]
    try:
        model(ids)
        return store["h"][0, -1, :]
    finally:
        for handle in handles:
            handle.remove()


def jacobian_for(ids, stage):
    """d h_final[last] / d h_stage[last]  ->  (hidden, hidden)"""
    outputs, prenorm_final = baseline_run(ids)
    injected = source_activation(outputs, prenorm_final, stage).requires_grad_(True)
    h_final = injected_run(ids, stage, injected)

    rows = []
    for start in range(0, hidden, CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, hidden)
        width = stop - start
        basis = torch.zeros(width, hidden)
        for offset in range(width):
            basis[offset, start + offset] = 1.0
        grads = torch.autograd.grad(
            outputs=h_final,
            inputs=injected,
            grad_outputs=basis,
            is_grads_batched=True,
            retain_graph=True,
        )[0]
        rows.append(grads[:, 0, -1, :])
    return torch.cat(rows, dim=0)


# ============================================================ PHASE 1
print("=" * 62)
print("PHASE 1 - VERIFICATION  (seconds, aborts on failure)")
print("=" * 62)

probe_ids = torch.tensor([tokenizer.encode(CORPUS[0])])
outputs, prenorm_final = baseline_run(probe_ids)

# Check A: does hidden_states[-1] equal norm(pre-norm final)?
# This settles empirically whether the last hidden state is normed.
renormed = model.model.norm(prenorm_final)
norm_gap = float((renormed - outputs.hidden_states[-1]).abs().max())
raw_gap = float((prenorm_final - outputs.hidden_states[-1]).abs().max())

print(f"  A. norm(prenorm_final) vs hidden_states[-1] : {norm_gap:.3e}")
print(f"     prenorm_final       vs hidden_states[-1] : {raw_gap:.3e}")
last_is_normed = norm_gap < raw_gap
print(f"     -> hidden_states[-1] is "
      f"{'NORMED (as expected)' if last_is_normed else 'NOT normed (unexpected)'}")
print()

# Check B: stage-30 Jacobian must be exactly the identity.
identity = torch.eye(hidden)
stage_n_jac = jacobian_for(probe_ids, n_layers)
identity_dev = float((stage_n_jac - identity).abs().max())
print(f"  B. stage {n_layers} Jacobian vs identity        : {identity_dev:.3e}")
print(f"     Frobenius norm {float(stage_n_jac.norm()):.2f} "
      f"(identity = {float(identity.norm()):.2f})")

if identity_dev > 1e-4:
    print()
    print("  ABORTING - stage 30 is not the identity. The injection or")
    print("  the h_final definition is still wrong. Do not fit.")
    sys.exit(1)

print("     -> PASS")
print()

# Check C: an intermediate stage should be non-trivial and not identity.
mid = n_layers // 2
mid_jac = jacobian_for(probe_ids, mid)
print(f"  C. stage {mid} Frobenius norm              : "
      f"{float(mid_jac.norm()):.2f}")
print(f"     deviation from identity              : "
      f"{float((mid_jac - identity).abs().max()):.3e}")
print("     -> should be clearly non-zero (a real transformation)")
print()
print("  All checks passed. Proceeding to fit.\n")


# ============================================================ PHASE 2
print("=" * 62)
print(f"PHASE 2 - FITTING  ({len(CORPUS)} prompts x {n_layers} layers)")
print("=" * 62)
print(f"Cache will be about {n_layers * hidden * hidden * 4 / 1e6:.0f} MB.\n")

accumulator = torch.zeros(n_layers + 1, hidden, hidden)
counts = torch.zeros(n_layers + 1)
overall_start = time.perf_counter()

for prompt_index, prompt in enumerate(CORPUS, start=1):
    ids = torch.tensor([tokenizer.encode(prompt)])
    prompt_start = time.perf_counter()

    for stage in range(1, n_layers + 1):
        accumulator[stage] += jacobian_for(ids, stage)
        counts[stage] += 1

    elapsed = time.perf_counter() - prompt_start
    total = time.perf_counter() - overall_start
    remaining = (total / prompt_index) * (len(CORPUS) - prompt_index)
    print(f"  [{prompt_index:2d}/{len(CORPUS)}] {elapsed:6.1f}s  "
          f"elapsed {total / 60:5.1f}m  "
          f"remaining ~{remaining / 60:5.1f}m   {prompt!r}")

jacobians = torch.zeros_like(accumulator)
for stage in range(1, n_layers + 1):
    jacobians[stage] = accumulator[stage] / counts[stage]

print(f"\nTotal fitting time: "
      f"{(time.perf_counter() - overall_start) / 60:.1f} min\n")


# ============================================================ FINAL CHECK
print("=" * 62)
print("POST-FIT CHECKS")
print("=" * 62)

final_dev = float((jacobians[n_layers] - identity).abs().max())
print(f"  Averaged stage {n_layers} vs identity : {final_dev:.3e}"
      f"   {'PASS' if final_dev < 1e-4 else 'FAIL'}")
print()
print("  Jacobian magnitude by stage (Frobenius norm, identity = 24.0):")
for stage in range(1, n_layers + 1):
    norm = float(jacobians[stage].norm())
    if stage <= 3 or stage >= n_layers - 2 or stage % 6 == 0:
        bar = "#" * int(min(norm * 1.5, 40))
        print(f"    stage {stage:2d}: {norm:8.2f}  {bar}")

torch.save(
    {
        "checkpoint": CHECKPOINT,
        "n_layers": n_layers,
        "hidden_size": hidden,
        "corpus_size": len(CORPUS),
        "corpus": CORPUS,
        "jacobians": jacobians,
        "convention": "d h_final_prenorm[last] / d h_stage[last], corpus average",
    },
    CACHE_PATH,
)
print(f"\nSaved to {CACHE_PATH}  "
      f"({CACHE_PATH.stat().st_size / 1e6:.1f} MB)")