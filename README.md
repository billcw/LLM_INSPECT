# LLM_INSPECT — Transformer Learning Lab v2.1

LLM_INSPECT is a local, browser-based teaching laboratory for examining how a
decoder-only transformer predicts and generates tokens. It runs
[`HuggingFaceTB/SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
on your CPU and exposes selected intermediate calculations without sending
prompts to an external inference service.

The central idea is simple: next-token prediction is the model's interface, not
a complete description of its learned internal computation. This lab lets a
learner inspect that interface, follow information through a transformer block,
perform controlled experiments, and see where interpretation becomes uncertain.

> **Teaching and experimentation only.** The lab is not a production inference
> server or a validated interpretability research platform. Model probabilities
> are not factual-confidence scores, and an attention pattern is not by itself a
> causal explanation.

## What you can examine

The interface is organized as nine lessons plus an Explorer mode:

1. **Tokens and sampling** — token positions and vocabulary IDs, full-vocabulary
   logits, stable softmax, temperature, probability, rank, and a sampled draw.
2. **One transformer block** — RMSNorm; Q, K, and V projections; grouped-query
   attention; RoPE; causal masking; weighted values; output projection; residual
   updates; and the gated MLP.
3. **Generate and trace** — seeded autoregressive generation, multiple runs,
   optional KV caching, EOS stopping, and the next-token distribution at every
   generated step.
4. **Compare prompts** — separate token axes, logit and probability changes,
   total variation, Jensen–Shannon divergence, and descriptive hidden-state
   comparisons.
5. **Attention patterns** — individual heads, mean attention, maximum envelopes,
   raw or relative views, exact cell inspection, and profile charts.
6. **Interventions** — zero-head and causal-uniform-attention interventions,
   last-position or all-position scope, controls, and full-distribution metrics.
7. **How weights learn** — actual shifted-target loss and perplexity without
   updating SmolLM2, plus a separate hand-checkable optimizer example.
8. **Lenses and probes** — raw and normalized layer states, the logit lens, an
   experimental local Jacobian-vector product, and an affine regression probe
   with separate calibration and held-out prompts.
9. **Verify and export** — runtime provenance, parameter and cache accounting,
   reconstruction checks, notes, and JSON exports.

The UI includes an AI/math glossary, a detailed explanation of pseudorandom
seeds, signed vector profiles, histograms, and matrix heatmaps. The graphs
summarize numeric arrays; exact values remain available. Adjacency or prominence
in a graph does not establish semantic similarity, importance, or causation.

For a detailed audit trail and learning guide, see
[`docs/Transformer_Lab_Changes_and_Learning_Guide.docx`](docs/Transformer_Lab_Changes_and_Learning_Guide.docx).
For the concise release history, see [`CHANGELOG.md`](CHANGELOG.md).

## Verification status

The following checks passed on the maintainer's Windows installation on
**September 8, 2026**, using the files on the v2.1 feature branch:

| Check | Result | What it establishes |
| --- | ---: | --- |
| `python -m unittest discover -s tests -v` | 46/46 passed | NumPy reference math plus model/API behavior |
| `python verify_model.py --output <fresh-path>` | 201/201 passed | Reconstruction and consistency checks against the downloaded SmolLM2-135M checkpoint |
| `node --test tests/test_math.js tests/test_ui.js` | 26/26 passed | Browser math and DOM-stub rendering behavior |

The Python suite emitted two dependency deprecation warnings: one from
Starlette/AnyIO and one for Transformers' legacy tuple-form KV cache. They were
warnings, not failed checks. The JavaScript UI tests use a small DOM stub rather
than a full browser, so they are not visual or end-to-end browser tests.

Earlier preparation also passed 52 offline mathematical and rendering tests;
[`verification_preparation.json`](verification_preparation.json) records that
specific environment and its limitations. These test results support the listed
calculations and software behavior. They do not prove that every visualization
has a unique interpretation or that findings generalize to other prompts,
models, revisions, devices, or dependency versions.

## Requirements

- 64-bit CPython **3.11 or 3.12**; Python 3.12 is the primary Windows path used
  for this release.
- A few gigabytes of free RAM and disk headroom. Exact peak resource use has not
  been benchmarked.
- Internet access during dependency installation and the first model load. The
  model can load from cache afterward.
- Node.js is optional and is needed only to rerun the JavaScript tests.

Dependencies are pinned in [`requirements.txt`](requirements.txt) as a tested
reference stack. The pins are not a claim that those versions are the newest or
security-current. Test upgrades before trusting measurements because framework
hooks, attention outputs, and cache interfaces can change.

## Windows setup with Python 3.12

Open PowerShell in the cloned repository:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

PowerShell activation is optional because the commands can address the virtual
environment's Python directly. If you prefer activation and local policy permits
it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Start the app on port 8000:

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 --workers 1
```

If another program already uses port 8000, choose a free port such as 8001:

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8001 --workers 1
```

Then open `http://127.0.0.1:8000` or the port you selected. Do not open
`static/index.html` directly with a `file://` URL. `run_windows.bat` uses port
8000; use the explicit command above when you need another port.

## Linux and macOS setup

On Linux with Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
bash run.sh
```

On macOS, install `torch==2.5.1` from the normal PyPI index instead of the
Linux/Windows CPU index, then install the remaining requirements. This release
has not been independently tested on macOS.

## Running the verification

With the virtual environment active, or by substituting its full Python path:

```powershell
python -m unittest discover -s tests -v
python verify_model.py --output "$env:TEMP\model_verification_v2_1.json"
node --test .\tests\test_math.js .\tests\test_ui.js
```

`verify_model.py` refuses to overwrite an existing report, so use a fresh output
path for each run. Exit status 0 and `PASS: 201/201` mean that report's checks
passed. Preserve a failed report and investigate the model revision, tokenizer,
normalization, indexing, backend, and dependency versions before changing any
tolerance.

The 20 model/API tests inside the 46-test suite use a tiny random, untrained
Llama model created locally. They test implementation behavior, not learned
knowledge. The 201-check verifier loads the actual pretrained checkpoint.

## Model and runtime

The default checkpoint is the **base**, not instruction-tuned, version of
SmolLM2-135M. The current upstream configuration describes a Llama-family causal
language model with 30 blocks, hidden size 576, vocabulary size 49,152, nine
query heads, three key/value heads, tied embeddings, and a configured maximum of
8,192 positions. The lab runtime reports 134,515,008 parameters for the loaded
checkpoint. See the upstream
[`config.json`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/main/config.json)
and [model page](https://huggingface.co/HuggingFaceTB/SmolLM2-135M).

The lab deliberately uses CPU, float32, eager attention, one Uvicorn worker, and
a shared lock around instrumented model operations. Temporary hooks inspect one
shared model; do not remove the lock when extending API routes.

Environment variables, set before starting the server:

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_LAB_MODEL` | `HuggingFaceTB/SmolLM2-135M` | Compatible checkpoint name or local path |
| `LLM_LAB_REVISION` | `main` | Requested Hugging Face revision |
| `LLM_LAB_OFFLINE` | unset | Set to `1` to use only local/cached files |
| `LLM_LAB_THREADS` | `4` | CPU thread count, clamped to 1–8 |

The interface reports the actual loaded configuration; use that runtime record
rather than assuming every alternate checkpoint matches the default dimensions.
For stronger reproducibility, pin and record a resolved checkpoint commit and
save `python -m pip freeze`. A random seed alone does not reproduce changes in
weights, software, hardware, numerical kernels, or sampling code.

## Workload limits

These are application safeguards, not the checkpoint's full context limit:

- Ordinary prompts: 256 tokens.
- Attention, inspection, intervention, comparison, and probe prompts: 64 tokens.
- Full verification prompts: 16 tokens.
- Generation: up to 64 new tokens and five runs.
- Prompt plus generation must fit the loaded model's context window.

Large attention payloads grow approximately with the square of sequence length.
Exports can therefore be sizable.

## Scientific boundaries

- **Attention is not automatically explanation.** Attention weights describe
  routing within one calculation. They omit value-vector content and downstream
  computation; a bright cell is a lead for investigation, not a causal finding.
- A maximum-across-heads display is an envelope, not a probability distribution.
- Dividing displayed weights by a uniform reference does not perform an
  intervention. The lab's separate causal-uniform operation does alter the
  forward calculation, but it is still an inference-time intervention—not a
  model trained with uniform attention.
- Norms, cosine similarity, MLP parameter share, and graphical magnitude are
  descriptive measurements. None alone measures meaning, knowledge, or
  intelligence.
- The affine state-regression probe is not the published KL-trained tuned lens.
  A local Jacobian-vector product is a derivative around one activation, not a
  global linear account of all later layers.
- A small held-out prompt set or a successful reconstruction check is not proof
  of general interpretability.
- Generated text and next-token probabilities are not guarantees of truth.

The interface links primary literature and labels externally reported results as
paper findings rather than measurements produced by this application.

## Privacy and security

The application performs inference locally and has no analytics or cloud
inference integration. Hugging Face is contacted to download model/tokenizer
files when they are not cached. Uvicorn's local access log may contain prompt
text used in the tokenization query. Exported JSON can contain prompts and notes.

Keep the server bound to loopback (`127.0.0.1`). It has no user authentication,
authorization, durable database, rate limiting for public use, or hardened
deployment configuration. Do not expose it to an untrusted network. Never load
untrusted Python files, checkpoints, or pickle-based caches.

## Repository map

| Path | Role |
| --- | --- |
| `main.py` | Validated local FastAPI routes, lazy engine, shared lock, and static serving |
| `engine.py` | Model execution, generation, instrumentation, interventions, probes, and verification |
| `math_core.py` | Independent NumPy calculations and hand-checkable examples |
| `static/index.html` | Lesson text and interface structure |
| `static/app.js` | Requests, state, rendering, and visualizations |
| `static/math.js` | Browser-side distribution and sampling math |
| `static/style.css` | Responsive presentation |
| `tests/` | Python math/model/API tests and JavaScript math/rendering tests |
| `verify_model.py` | Actual-checkpoint reconstruction and consistency report |
| `verification_preparation.json` | Historical preparation-environment test record |
| `docs/` | Learning guide and project-history material |
| `archive/v1_experiments/` | Unsupported version-1 exploratory scripts retained for history |
| `originals/` | Byte-preserved source files supplied for the v2 reconstruction |

The v2 API is intentionally not backward-compatible with version 1. Heavy
operations use POST JSON, `/api/intervene` replaces the old knockout endpoint,
and `/api/lenses` no longer returns the old averaged-Jacobian format. Interactive
API schemas are available at `/docs` while the server is running.

## AI-assisted development and human responsibility

Bill (`billcw`) conceived the learning goal, directed the project, selected and
evaluated features, tested the program against the actual model, challenged
claims, chose what to retain, and made the release decisions.

The original implementation was developed with substantial assistance from
Anthropic's Claude; most of the original Python and JavaScript was produced in
those interactive sessions. OpenAI Codex performed a later scientific and code
audit and substantially reconstructed version 2, including additional
instrumentation, mathematical checks, tests, documentation, safety boundaries,
and the version 2.1 learner-interface improvements. See
[`ACKNOWLEDGMENTS.md`](ACKNOWLEDGMENTS.md) for the fuller account.

AI-generated material was treated as draft work requiring human direction,
selection, testing, and correction. Acknowledgment describes the development
process; it does not transfer responsibility for the published repository to an
AI system or imply endorsement by Anthropic or OpenAI.

## License

Project code and documentation are released under the [MIT License](LICENSE):
use, copy, modify, merge, publish, distribute, sublicense, or sell copies, subject
to preserving the copyright and license notice. The software is provided without
warranty.

The downloaded SmolLM2 checkpoint is a separate work supplied by Hugging Face
under its own [Apache-2.0 license](https://huggingface.co/HuggingFaceTB/SmolLM2-135M).
Python and JavaScript dependencies retain their respective licenses. The MIT
license for this repository does not replace those third-party terms.
