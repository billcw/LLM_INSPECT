"""
token_probe.py - inspect how the tokenizer handles a string.
BPE merge order correlates with corpus frequency, so a low token ID
means the pattern was common in the data the tokenizer was built on.
"""

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

for text in ["donot", "do not", "don't", "don", " ot", "ot", "cannot", "pseudonotum"]:
    ids = tok.encode(text)
    pieces = [tok.decode([i]) for i in ids]
    print(f"{text!r:10s} -> {len(ids)} tokens  "
          f"ids={ids}  pieces={pieces}")