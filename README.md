# LLM_INSPECT

A local, browser-based teaching tool for showing what a language model actually
does when it predicts the next token. It runs a small model on your own CPU and
exposes the intermediate steps — tokenization, the full logit vector, temperature
and sampling, the residual stream read out layer by layer, and attention weights
for every layer and head.

It was built to answer a specific objection: that a language model is "just
guessing." The tool tries to make the computation legible enough that a
non-specialist can see it is structured, deterministic given its inputs, and
inspectable — and can also see exactly where inspection stops working.

## A note on how this was built

I am not a machine learning engineer. I built this to teach myself how
transformers work and to demonstrate it to others, and I had substantial help
from Claude (Anthropic) throughout — most of the Python and JavaScript here was
written by Claude in conversation with me, across many sessions.

What I contributed was direction, testing, and verification: I ran every claim
against the actual model on my machine, and a number of statements that started
out in this tool turned out to be wrong and were removed or corrected as a
result. The project record (`llm-demo-project-record-v3.md`) includes a section
that logs claims that were made and later disproven, several of them mine and
several of them Claude's. That log is deliberately part of the repository.

I'm noting this because I would rather be accurate about it than have anyone
assume a level of expertise I don't have. If you're evaluating my skills from
this repo: the understanding is real and hard-won, the code is collaborative.

## What it shows

- **Tokenization** — live token chips as you type, so the "words are not tokens"
  point lands before anything else.
- **The full distribution** — the complete logit vector over all 49,152 tokens is
  sent to the browser once per prompt, so temperature can be applied client-side.
  The ranking is by logit, which is invariant to temperature.
- **Sampling as a dart throw** — the probability distribution laid out on a number
  line, with the sampled draw shown as a point on it.
- **Generation with tracing** — step by step, showing the context growing and the
  candidates at each step. Run it several times at temperature 0 to see that the
  model is deterministic; raise the temperature to see runs diverge.
- **Two lenses on 30 layers** — the logit lens (read the prediction out early) and
  a self-fitted Jacobian lens (approximate the remaining layers with one averaged
  linear map). Stage 30's Jacobian is the identity, so it must reproduce the real
  logits exactly; that check is displayed as a pass/fail diagnostic.
- **Attention** — weights for all 30 layers and all 9 query heads, with a
  `× uniform` scale that divides each cell by what uniform attention would give
  for that row, plus a sink-profile chart across all layers.

## What it deliberately does not claim

The attention section carries an explicit panel about its own limits. Attention
weights are half the operation — a head weights positions and then averages their
*value* vectors, and the grid shows only the weights. A bright cell is a
hypothesis, not a finding.

That framing follows a specific literature: Jain & Wallace, *Attention is not
Explanation* (NAACL 2019); Serrano & Smith, *Is Attention Interpretable?* (ACL
2019); and the rebuttal, Wiegreffe & Pinter, *Attention is not not Explanation*
(EMNLP 2019). The `× uniform` control turns out to implement the uniform-weights
baseline that the rebuttal recommends. All three papers studied text classifiers
with a single attention layer rather than 30-layer decoders, and the tool says so
rather than borrowing their authority.

The attention-sink behaviour visible in the tool is consistent with StreamingLLM
(Xiao et al., ICLR 2024), which is cited in the interface.

## Requirements

- Python 3.14 (developed on 3.14.7)
- CPU only — no GPU needed. The model is ~538 MB in fp32.
- Roughly 2 GB free disk for the model download on first run.

Developed on Windows 11 with `cmd.exe`. Nothing here is Windows-specific, but the
commands below are.

## Setup

```
git clone https://github.com/billcw/LLM_INSPECT.git
cd LLM_INSPECT

python -m venv llmdemo
.\llmdemo\Scripts\activate.bat

pip install -r requirements.txt
```

The model (`HuggingFaceTB/SmolLM2-135M`, Apache 2.0) downloads automatically from
Hugging Face the first time you start the server.

## Running

```
uvicorn main:app --reload --port 8000
```

Then open <http://127.0.0.1:8000>.

The Jacobian lens needs `jacobian_cache.pt`, which is **not** in this repository —
it is about 41 MB and is a build artifact. Generate it once:

```
python fit_jacobian.py
```

Everything else works without it; the lens section reports that it is unavailable
rather than failing.

## The model

`HuggingFaceTB/SmolLM2-135M` — 134,515,008 parameters, 30 layers, hidden size
576, vocabulary 49,152, 9 query heads and 3 key/value heads (grouped-query
attention), RMSNorm, RoPE, tied embeddings. It is small enough to run
comfortably on a laptop CPU and large enough to show real structure.

The model is loaded with `attn_implementation="eager"`, which is required for
`output_attentions=True` — the default fused backend never materializes the
weight matrix and returns an empty tuple instead.

## Repository contents

| File | What it is |
| --- | --- |
| `main.py` | FastAPI backend. Tokenize, predict, generate, lenses, attention. |
| `static/index.html` | The entire frontend — one page, no build step. |
| `fit_jacobian.py` | Offline fitting that produces `jacobian_cache.pt`. |
| `jacobian_probe.py` | Exploratory script used while working out the fitting. |
| `llm-demo-project-record-v3.md` | Working record: what was built, what was verified, what was disproven, what is still open. |

## Known limitations

- Generation deliberately recomputes the full sequence each step with no KV
  cache. It is slower, and the computation stays obvious.
- The attention payload grows with the square of the token count and is capped at
  32 tokens.
- The Jacobian lens is a linear approximation of a non-linear stack, fitted on a
  12-prompt corpus. It is a teaching device, not a research instrument.
- Findings recorded in the project record were measured on specific prompts and
  should not be read as general properties of the model. The interface now
  computes its numbers live from whatever prompt you type, for exactly this
  reason.

## License

The code here has no license applied yet — add one before treating it as
reusable. The model is Apache 2.0, licensed by Hugging Face, and is downloaded
rather than redistributed here.
