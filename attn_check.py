"""
attn_check.py - why did output_attentions=True return an empty tuple?

Compares the default attention implementation against 'eager'.
Loads the model twice, so expect roughly 1.1 GB peak. Stop the uvicorn
server first if memory is tight.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CHECKPOINT = "HuggingFaceTB/SmolLM2-135M"
PROMPT = "The capital of Japan is called"

torch.set_grad_enabled(False)

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
ids = torch.tensor([tokenizer.encode(PROMPT)])
print(f"prompt: {PROMPT!r}  ({ids.shape[1]} tokens)\n")

for impl in [None, "eager"]:
    kwargs = {} if impl is None else {"attn_implementation": impl}
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT, dtype=torch.float32, **kwargs)
    model.eval()

    outputs = model(ids, output_attentions=True)
    attn = outputs.attentions

    print(f"requested impl : {impl!r}")
    print(f"  config reports : {model.config._attn_implementation!r}")
    print(f"  attentions is None : {attn is None}")
    print(f"  len(attentions)    : {0 if attn is None else len(attn)}")

    if attn is not None and len(attn) > 0:
        first = attn[0]
        print(f"  entry 0 shape      : {tuple(first.shape)}")
        print(f"  last row sums to   : {float(first[0, 0, -1, :].sum()):.6f}"
              f"   (must be 1.0)")
        print(f"  upper-triangle max : "
              f"{float(first[0, 0, 0, 1:].abs().max()):.3e}"
              f"   (causal mask - must be 0)")
    print()

print("If 'eager' returns 30 entries and the default returns 0, the fix is")
print("to pass attn_implementation='eager' when loading the model in main.py.")