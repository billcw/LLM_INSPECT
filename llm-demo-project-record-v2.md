# LLM Next-Token Demo — Complete Project Record (v2)

**Purpose:** a local, inspectable demonstration of how a language model predicts
the next token, built to show skeptical coworkers that the process is
structured computation rather than guesswork.

**Machine:** HP EliteBook, Windows 11, CPU only. No GPU used at any point.

**Status:** build complete. All six UI sections built and verified. Remaining
work is documentation and demo preparation, not code — see §13.

*Supersedes v1. New in this version: rank lookup, client-side temperature,
number-line sampling widget, attention view, and four bugs found and fixed
(§8). Corrections to v1 are marked.*

---

## 1. Environment

Verified from the running system, not assumed:

| Component | Version |
|---|---|
| Python | 3.14.7 |
| torch | 2.13.0+cpu |
| transformers | 5.16.1 |
| fastapi | 0.141.1 |
| uvicorn | 0.52.4 |

**Paths**

- Project root: `C:\Projects\llm-demo`
- Virtual environment: `C:\Projects\llm-demo\llmdemo`
- Static files: `C:\Projects\llm-demo\static\index.html`
- HuggingFace cache: `C:\Users\billc\.cache\huggingface\`

**Activation** (cmd.exe, not PowerShell):

```
cd C:\Projects\llm-demo
.\llmdemo\Scripts\activate.bat
```

**Run:**

```
uvicorn main:app --reload --port 8000
```

Then `http://127.0.0.1:8000/`

**Expected startup lines:**

```
Model loaded. Attention implementation: eager
Jacobian cache loaded (12 prompts).
```

Both matter. If the first says `sdpa`, attention breaks (§8.3). If the second
is missing, the J-lens column is empty.

### Environment notes learned the hard way

- **`Activate.ps1` is a PowerShell script.** In cmd.exe it silently does
  nothing — no error, no `(llmdemo)` prefix. Use `activate.bat`.
- **Python 3.14 is very new.** PyTorch supports it on Windows, but CUDA wheels
  don't exist for 3.14 yet — CPU only. Convenient here, not a limitation.
- **`--reload` restarts on file changes only.** Ctrl-C kills the server. Use a
  second cmd window for other commands.
- **`--reload` also caused a real bug** — see §8.4.
- **Static files don't trigger `--reload`.** Editing `index.html` requires a
  browser hard-refresh (Ctrl+F5).
- **`torch_dtype` is deprecated** in transformers 5.16.1. Use `dtype=`.
- **`favicon.ico 404`** in the log is harmless.

---

## 2. The model: SmolLM2-135M

Read directly from the loaded model:

```
model_type                 = llama
num_hidden_layers          = 30
hidden_size                = 576
intermediate_size          = 1536
num_attention_heads        = 9
num_key_value_heads        = 3        (grouped-query attention)
vocab_size                 = 49152
max_position_embeddings    = 8192
tie_word_embeddings        = True
rms_norm_eps               = 1e-05
rope_parameters            = {'rope_theta': 100000, 'rope_type': 'default'}

parameter count            = 134,515,008
fp32 weight size           = 538.1 MB
lm_head == embed_tokens    = True
```

### Version trap: SmolLM vs SmolLM2

Configs look nearly identical. The two distinguishing fields, confirmed here:

| Field | v1 | **v2 (this model)** |
|---|---|---|
| `rope_theta` | 10000 | **100000** |
| `max_position_embeddings` | 2048 | **8192** |

Layers, hidden size, vocab, and tied embeddings are identical between versions
and cannot be used to tell them apart.

### Architecture differences from GPT-2 that shaped the build

1. **30 layers, 576 dims** (not 12 × 768). Taller lens table.
2. **Tied embeddings** — the unembedding matrix *is* the embedding matrix.
3. **RMSNorm, not LayerNorm.** Final norm at `model.model.norm`, not `ln_f`.
4. **RoPE, not learned position embeddings.** Position is injected by rotation
   inside attention, so the embedding row of the residual stream is *pure token
   identity*, position-free.
5. **GQA: 9 query heads, 3 KV heads.** Attention weights are per query head.

### Vocabulary layout (measured, §7.3)

- IDs **0–16**: 17 special tokens
- IDs **17–272**: 256 byte tokens
- IDs **273+**: BPE merges, in frequency order

---

## 3. Core concepts

### 3.1 What the model does

One thing: given text, produce a probability for every one of the 49,152 tokens
in its vocabulary. It does not plan a sentence, look anything up, or have
memory between runs.

**The guessing baseline:** `1 / 49,152 = 0.00203%`. Every probability observed
should be compared against this.

### 3.2 Tokens, not words

Text is split by BPE into chunks from a fixed vocabulary. Common words are
usually one token; rare words are split. Most tokens carry a **leading space**:
`" capital"` ≠ `"capital"`.

When `" Os"` or `" Sh"` appears in predictions, the model is starting to spell
*Osaka* or *Shizuoka* and will finish next pass. It predicts tokens, not words.

### 3.3 The grid: positions × layers

A transformer does not process a sentence as one object. **Every token gets its
own 576-number vector**, and those travel through the 30 layers *side by side,
in parallel*.

- Columns = token positions (set by the tokenizer)
- Rows = 31 stages (embeddings + 30 layer outputs)

A 6-token prompt = 6 × 31 = **186 separate 576-number vectors**.

The prediction is read from one cell: **top row, rightmost column**. Every other
cell feeds in indirectly through attention.

### 3.4 The residual stream — vectors are added to, not replaced

**The biggest conceptual correction in the project.**

Each layer does:

```
h = h + attention(norm(h))
h = h + mlp(norm(h))
```

Each sublayer computes a **correction** and adds it. It does not overwrite.
30 layers × 2 sublayers = **60 additions** on the original embedding.

```
h_final = embedding + (60 corrections summed)
```

The embedding vector is still present at layer 30. The layers didn't replace
the token's meaning; they annotated it with context.

**Measured:** `|h|` grows from **2.5** at embeddings to **729.1** at layer 29 —
roughly **292×**. The apparent drop to 44.2 at stage 30 is HuggingFace's
`hidden_states[-1]` having had the final RMSNorm applied, which rescales it.

### 3.5 Causal masking

Every position attends to itself and positions *before* it, never after.

Tested directly:

- **Appending** tokens leaves earlier activations unchanged: max abs difference
  **1.526e-05**
- **Prepending** changes them completely: **1.245e+01**

Six orders of magnitude apart. The append difference is floating-point rounding
noise — BLAS can choose different summation orders for different matrix shapes,
and floating-point addition is not associative.

Independently confirmed in the attention view: **upper-triangle max is exactly
0.000e+00**. Not small — zero.

This is why KV caching works.

### 3.6 The forward pass

One trip through the model: tokens in, 49,152 logits out. No loops, no
revision. (The *backward* pass only happens during training.)

1. Each token → embedding table → 576-number vector
2. Vectors travel through 30 layers side by side
3. Last position's final vector → 49,152 scores

Two properties that surprise people:

- **Nothing is stored between passes.** It "remembers" earlier text only
  because that text is fed in again as input.
- **Computation is fixed.** Easy and hard prompts cost the same. The model
  cannot think longer about a harder question.

Measured: **61.1 ms** for a 6-token prompt.

### 3.7 Unembedding — 576 numbers become 49,152 scores

The bridge is a matrix of shape **49,152 × 576**.

**Each row is a 576-number vector representing one vocabulary token.** The
multiplication is 49,152 dot products:

```
logit for " Tokyo" = dot( h_final , row_for_Tokyo )
```

A dot product measures alignment. The final step asks, 49,152 times: *how much
does the vector I built point the same direction as this token's vector?*

**Tied embeddings:** the unembedding matrix **is** the embedding matrix — the
same table, both directions.

- Read one row → token becomes a vector (start)
- Dot against all rows → vector becomes token scores (end)

**Empirical confirmation:** the embeddings row of the lens table reads
`called 100.0%` when the last input token is `" called"`. With tied embeddings
that vector *is* that token's row, so dotting it against all rows returns
itself at maximum.

### 3.8 Logits, softmax, and temperature

**Logits** are 49,152 unbounded scores. Not probabilities.

**Softmax:** exponentiate each, divide by the total.

**Temperature** divides the logits first: `p = softmax(logits / T)`.

- Small T → concentrates
- Large T → flattens
- T = 0 → collapses to argmax (greedy)

**The logits never change with temperature.** The model has already finished.

**Ranking never changes either.** Dividing every logit by the same positive
number is a *monotonic* transformation, so the top-k list is identical in
content and order at every T > 0. Only the probabilities move.

Useful identity: `p_T ∝ p_1^(1/T)`.

### 3.9 How a token is selected — the number line

The "raffle ticket" intuition is essentially correct. The algorithm:

1. Softmax the logits → 49,152 probabilities summing to 1
2. **Cumulative sum** → boundaries on a line from 0 to 1, each token occupying
   a segment as wide as its probability
3. Draw one uniform random number `r` in [0, 1)
4. Take the first token whose boundary exceeds `r`

Worked example from an actual run:

```
  0.0000 -> 0.2780   know    (width 0.278)
  0.2780 -> 0.3840   want    (width 0.106)
  0.3840 -> 0.4710   like    (width 0.087)
  0.4710 -> 0.5490   think   (width 0.078)
  0.5490 -> 0.6050   have    (width 0.056)
  0.6050 -> 1.0000   [49,147 other tokens]
```

`r = 0.15` → `know`. `r = 0.42` → `like`. `r = 0.95` → deep in the tail.

**Why the number line beats physical tickets:** a token at probability 0.000001
would need a million tickets. On the line it gets a segment 0.000001 wide.

**Temperature resizes the segments** before the dart is thrown. At T = 0.5 the
top segment above grows from 27.8% to 73.4% of the line.

**At T = 0 there is no dart.** `argmax` takes the widest segment; the PRNG is
never called.

### 3.10 The random number knows nothing

The PRNG has never seen the prompt, logits, tokens, or model. One line:

```python
next_id = int(torch.multinomial(sample_probs, num_samples=1))
```

That line would work identically on a weather model or a hand-typed list of
five probabilities.

**But the number line is not independent.** The model decides *what the
segments are*. The random number decides *where the dart lands*. The dart is
blind; the board is not.

Precise phrasing: `r = 0.15` is not a choice about tokens. It becomes `know`
only because the model put `know` at 0.0–0.278. Change the prompt and the same
`r` lands somewhere entirely different.

**Seeds:** fixing the seed fixes the sequence of `r` values, making sampled
output exactly reproducible.

### 3.11 Autoregression

The model predicts one token. Generation is a loop:

1. Tokenize the prompt
2. Forward pass → 49,152 logits
3. Pick one token
4. **Append it to the input**
5. Return to step 2

12 tokens = 12 complete forward passes. Nothing is planned ahead.

**This explains many failure modes.** Once a wrong token is appended it is in
the context permanently. The model cannot revise it.

---

## 4. The two lenses

### 4.1 Logit lens

```
readout(h_L) = lm_head(norm(h_L))
```

Take the residual stream after layer L, apply the same final normalization and
unembedding matrix, see what token it points at.

**Weakness:** it pretends layers L+1 through 30 don't exist.

**Deeper assumption:** the unembedding matrix was trained to read the *final*
layer. Applying it to layer 6 assumes a consistent reading frame throughout.

### 4.2 Jacobian lens

Anthropic's J-lens (released July 2026, paper: *Verbalizable Representations
Form a Global Workspace in Language Models*) adapts the logit lens:

```
readout(h_L) = lm_head(norm(J_L · h_L))
J_L = E[ ∂h_final / ∂h_L ]      averaged over a corpus
```

*If I nudge this number in the stream at layer L, how does the final vector
move, on average?* One matrix multiply stands in for layers L+1 through 30.

**Cost difference:** the unembedding matrix already exists in the weights, so
the logit lens is free. The Jacobians must be computed with autodiff over a
corpus and cached.

**Caveat on this implementation:** Anthropic's expectation runs over current
*and future* target positions. This implementation takes the same-position
slice (∂h_final[last] / ∂h_L[last]). It is a J-lens in the same spirit, **not a
reproduction of theirs**. Their implementation was not read.

### 4.3 How the Jacobians were computed

**Forward hook** technique:

1. Run normally, capture the residual stream at layer L
2. Copy it, mark `requires_grad=True`
3. Hook swaps the copy in at layer L
4. Run again — `h_final` now connects through the autograd graph
5. Each backward pass yields **one row** of the 576 × 576 Jacobian

**Important subtlety:** `model.requires_grad_(False)` is set and does *not*
break this. We differentiate with respect to an *injected activation*, not the
weights. Parameter gradients and activation gradients are separate things;
disabling the former just makes the graph smaller and the fit faster.

**Timing measured:**

| Path | per row | one Jacobian | 30 layers × 10 prompts |
|---|---|---|---|
| Sequential backward | 24.4 ms | 14.0 s | 1.2 hours |
| Batched (`is_grads_batched=True`, vmap) | 4.5 ms | 2.6 s | ~13 min |

Batched is **5.4× faster**. Actual fit: **22.8 minutes**, 12 prompts, **41.1 MB**
cache.

### 4.4 The free correctness check

At stage 30 the Jacobian is the identity by definition, so the J-lens **must**
reproduce the model's real logits there.

- Stage 30 Jacobian vs identity: **0.000e+00**
- Frobenius norm: **24.00** (identity of dim 576 = √576 = 24.0)
- J-lens output vs real logits at stage 30: **2.289e-05, PASS**

---

## 5. Files

| File | Purpose |
|---|---|
| `probe.py` | Environment, config, tokenization, top-k, causal-mask test |
| `main.py` | FastAPI server (6 endpoints + static mount) |
| `static/index.html` | Single-page UI with inline explanations |
| `jacobian_probe.py` | Timing probe — measures Jacobian cost before committing |
| `fit_jacobian.py` | Two-phase: verify in seconds, then fit and cache |
| `jacobian_cache.pt` | 41.1 MB cached Jacobians (12-prompt corpus) |
| `token_probe.py` | Tokenizer inspection — splits and token IDs |
| `attn_check.py` | Compares default vs eager attention implementations |
| `llm-demo-project-record.md` | This file |

### Endpoints

| Endpoint | Forward passes | Note |
|---|---|---|
| `/api/tokenize` | **0** | tokenizer only; fires on every keystroke |
| `/api/decode` | **0** | single token id → text |
| `/api/predict` | 1 | returns the COMPLETE 49,152 logit vector |
| `/api/generate` (n=12, runs=5) | 60 | ~3.7 s |
| `/api/lenses` | 1 | but 61 unembedding readouts ≈ 1.73 billion MACs |
| `/api/attention` | 1 | returns 30 × 9 = 270 grids |
| `/api/health` | 0 | reports attn implementation and J-lens availability |

---

## 6. UI sections

1. **The prompt** — live tokenization while typing
2. **Next-token distribution** — logits, temperature slider, rank lookup, and
   the number-line dart widget
3. **Generation** — n tokens, runs, seed; determinism demonstration
4. **Step-by-step trace** — one block per forward pass, with chosen-token rank
5. **Two lenses on 30 layers** — logit lens and J-lens side by side
6. **Attention** — 270 grids, layer and head selectors

### Client-side temperature

`/api/predict` returns all 49,152 logits (~390 KB, rounded to 3 decimals). The
browser recomputes the softmax when the slider moves.

**Zero server requests when dragging the slider.** This is a demo feature, not
just an optimization: *"watch the server log while I change temperature"* and
nothing appears — because the model already finished.

Full logits are needed (not just top-k) because an exact softmax requires the
complete denominator.

---

## 7. Empirical findings

### 7.1 Prediction quality

**"The capital of Japan is"**

| token | prob | × uniform |
|---|---|---|
| ` Tokyo` | 18.28% | 8,985× |
| ` the` | 13.07% | 6,424× |
| ` located` | 6.14% | |
| ` Kyoto` | 3.31% | 1,627× |

**"The capital of Japan is called"** — one word nearly doubled confidence:

| token | prob | × uniform |
|---|---|---|
| ` Tokyo` | **36.22%** | **17,803×** |
| ` Kyoto` | 6.32% | |
| ` N` | 6.01% | |
| ` As` | 5.93% | |
| ` the` | 3.78% | (dropped from rank 2 to rank 5) |

Top 10 tokens (0.02% of vocabulary) hold **69.8%** of total probability.

**"The capital of France is"** — weaker: `Paris` **second** at 9.38%, behind
`the` at 26.17%. Lead with Japan.

**"test"** (bare token) — top 13 predictions are almost entirely code
punctuation: `_`, `.`, `(`, `)`, `\n`, `,`, `()`, `>`, `:`, `;`, `::`, `=`.
The corpus contains enough code that a bare `test` reads as an identifier, not
an English word. Useful demo of how training data shapes predictions.

### 7.2 The 20-run structured-failure audit

Twenty samples at T = 1.0, "The capital of Japan is called ___":

| category | count |
|---|---|
| `Tokyo` | 10 / 20 |
| Real Japanese place (Tokyo, Nara ×2, Kyoto ×2, Kyushu, Fushimi) | **16 / 20** |
| Fabricated but Japanese-phonetic (Japhethius, Agivao, Yamato-Komusubutsu) | 3 / 20 |
| Non-Japanese place | **0 / 20** |

Nara and Kyoto are *actual former capitals of Japan*.

**Necessary qualification:** the *answer slot* never leaves Japan. The
*continuation text* does — "Pearl River Delta" (China) and "Korean money"
appear after the answer. Correct claim: *the direct answer stays in Japan; the
text after it degrades.*

### 7.3 The `ot` token investigation

Question: why does `ot` appear in the top 10 after "I don"? Typos of "donot"?

```
'donot'       -> 2 tokens  ids=[12420, 320]     pieces=['don', 'ot']
'cannot'      -> 1 token   ids=[33551]
'pseudonotum' -> 4 tokens  ids=[47438, 258, 320, 382]
                            pieces=['pseud', 'on', 'ot', 'um']
```

**Decisive test:** `ot` is the *same token, id 320*, in both `donot` and
`pseudonotum`. The latter has nothing to do with "do not." So `ot` is a
general-purpose subword fragment, not a "donot" artifact.

**What id 320 means:** merges begin at 273 (§2), so id 320 is merge **#47** —
roughly the 47th merge BPE ever made, top 0.1%. "ot" is one of the most common
two-character sequences in English (*not, got, hot, lot, spot, foot, rot*).

*Correction to v1: the merge offset was estimated as 272, making this #48. The
measured value is 273 and #47. The argument is unaffected.*

**Strongest counter-evidence to the typo story:** `cannot` is a single token —
BPE merged it. `donot` never did. If "donot" typos were common enough to
matter, BPE would likely have merged it too.

**Confound:** ID order conflates frequency with *length*. Only same-length
comparisons are sound.

**Verdict:** `ot` is there because "ot" is an extremely common letter pair, not
because people misspell "do not."

### 7.4 Vocabulary layout, measured

From the raw logit array for the prompt `test`:

- Indices **0–16** include **eleven consecutive entries of exactly 9.97**.
  Identical logits to three decimals across eleven tokens means special tokens
  whose embeddings were barely touched during training. 17 specials total.
- Index **17** jumps to 25.685 and stays in the 23–30 band — the 256 byte
  tokens.
- Merges therefore begin at **273**.

That eleven-way tie is also the direct cause of the original T=0 display bug
(§8.1) — those were exactly the tokens `topk` surfaced when everything tied.

### 7.5 Determinism

- T = 0, 5 runs: **all identical**
- T = 1.0, 10 runs: **10 distinct outputs**
- T = 1.0 with fixed seed: identical across runs

At T = 0.10, five runs produced identical text for **eleven consecutive
tokens**, then split at a paragraph boundary where several sentence-openers had
comparable probability. A clean illustration of the number line.

### 7.6 Lens comparison

| | reaches top-5 | reaches #1 |
|---|---|---|
| **Logit lens** | stage 30 | stage 30 |
| **Jacobian lens** | **stage 24** | **stage 24** |

The logit lens **never finds the answer early.** From layer 1 through layer 23
it sits between rank #8,000 and #14,000 while reporting commas and periods at
90%+ confidence. It drops to #413 at layer 24 and reaches #1 only at the end.

The J-lens reaches #1 at layer 24 and **holds it for seven consecutive layers**.
Layers 22–23 visibly assemble the *concept* before the answer:

- Layer 22: `Hare 2.5% Constantinople 1.9% Capital 1.6%` → rank #47
- Layer 23: `Capital 4.9% City 4.8% Cities 4.4%` → rank #10
- Layer 24: `Tokyo 72.1% Japan 25.9%` → rank #1

**29 of 31 stages disagree** on the top token.

**Jacobian Frobenius norms by stage** (identity = 24.0), non-monotonic with no
confident interpretation:

```
stage  1: 24.97    stage 18: 25.88
stage  2: 21.48    stage 24: 35.59
stage  3: 19.57    stage 28: 31.41
stage  6: 17.19    stage 29: 30.00
stage 12: 21.72    stage 30: 24.00
```

### 7.7 Attention

Layer 30, mean of all 9 heads, "The capital of Japan is called":

```
          The  capital  of  Japan  is  called
The       100
capital    79      21
of         75             20
Japan      66                   21
is         62                        25
called     65                        11     14
```

**The causal mask is a clean triangle.** Rows sum to 1. Upper triangle exactly
zero.

**Column 1 is the attention sink.** Every position sends 62–100% of its budget
to `The` — the first token — with nothing about that word warranting it. Much
stronger than expected on a 6-token prompt.

Consequence: the bottom-row summary (`The #0 65.4%, called #5 14.3%, is #4
11.3%`) is mostly reporting the sink. The interesting number is `called` at
14.3% — the previous token.

**The Japan→France rerouting demo cannot work at layer 30 with mean-of-9.**
With 65% of the budget on position 0, there isn't enough left to show
rerouting. Middle layers (12–20) with individual heads are the place to look.
Whether any head shows a clean switch on this model is **unverified**.

### 7.8 Temperature recovery from a screenshot

A displayed table showed `'t` at 66.01% while the summary reported "48748×
more likely than chance" (= 99.18% raw). Those disagree because the probability
column is temperature-adjusted and the summary uses raw T=1 values.

Solving from the top-2 logit gap of 5.43: **T ≈ 1.75**.

Consequence: `ot` displayed at 0.4357% is actually **0.0149%** at T = 1 — about
**7× uniform**, the noise floor of a distribution where the winner takes 99.18%.

*This motivated the current UI, which labels the probability column with the
active T and always reports summary statistics at T = 1.*

---

## 8. Errors made and corrected

Kept deliberately — the corrections are part of the record.

1. **"Exactly identical" on the causal-mask test.** Predicted exact zero; got
   1.526e-05. Correct phrasing is "identical up to floating-point rounding."
   The underlying claim held — the six-order-of-magnitude gap versus prepending
   is what confirms it.

2. **T = 0 top-k tie-break bug.** Ranking by temperature-adjusted probabilities
   at T = 0 produced a 49,151-way tie broken by vocabulary index, displaying
   `<gh_stars>`, `<reponame>` and similar as apparent predictions. **Fix:**
   always rank by raw logits; use adjusted values for display only. §7.4
   identifies the exact tokens involved.

3. **`torch_dtype` vs `dtype`.** Deprecated in transformers 5.16.1.

4. **Contradictory instructions** for placing `app.mount` — "near the top" and
   "at the very end" in the same paragraph. Correct rule: the mount catches all
   unmatched paths, so it must come *after* the API route definitions.

5. **Sanity check placed after the expensive operation.** The first
   `fit_jacobian.py` ran 22.5 minutes and *then* reported failure. Rewritten as
   two phases: verify in seconds, abort on failure, fit only if passing.

6. **The `hidden_states` norm bug.** `hidden_states[30]` has already had the
   final RMSNorm applied; `hidden_states[1..29]` have not. Injecting the normed
   tensor where the unnormed one belongs produced a stage-30 Jacobian that was
   the *RMSNorm* Jacobian (max deviation 2.116, Frobenius 22.88 vs 24.0)
   instead of the identity. **Fix:** capture the pre-norm output of the last
   decoder layer by hook.

7. **Stale caveat about `torch.set_grad_enabled(False)`.** Raised before the
   fit-offline-and-cache design was chosen, then never retracted. Another AI,
   consulted independently, read the stale warning back as a live concern. It
   was moot: `fit_jacobian.py` never disables grad globally, and `main.py`
   never computes a Jacobian.

8. **Wrong prediction about lens convergence.** Predicted disagreements would
   "cluster in the middle layers and vanish near the end." Actual: 29 of 31
   rows disagree, persisting through layer 29.

9. **Over-strong claim about structured failure.** Initially stated every
   T=1.0 error stays inside the Japanese semantic neighborhood. True for the
   answer slot (0/20), but the continuation text drifts.

10. **Stale tokenization display.** Token chips only refreshed on Enter or
    slider release, so typing a new prompt left old tokens on screen. **Fixed**
    by adding `/api/tokenize`, which runs the tokenizer without the model and
    fires on a 250 ms debounce while typing.

11. **SDPA returns no attention weights.** `output_attentions=True` returned an
    *empty tuple* rather than raising, producing a cryptic
    `Cannot read properties of undefined` error in the UI. Cause: PyTorch's
    fused scaled-dot-product-attention never materializes the weight matrix —
    that's the optimization. transformers stated it plainly once the right
    diagnostic was run: `sdpa attention does not support output_attentions=True`.
    **Fix:** load with `attn_implementation="eager"`, plus a server-side guard
    returning an explanatory error instead of an empty list.

    *Worth keeping as a demo point: the fast path is fast precisely because it
    throws away intermediate results. Production runs SDPA; this tool runs eager
    because we want to see inside. A real cost of interpretability.*

12. **`requires_grad` leaking under `--reload`.** `torch.set_grad_enabled(False)`
    runs at import, but under WatchFiles re-imports the global state did not
    hold, and logits came back carrying an autograd graph. Numbers were correct
    (gradient tracking doesn't change the forward computation) but memory and
    time were wasted on every request. **Fix:** `@torch.no_grad()` on all four
    model-touching functions, which applies per-call regardless of import order.

    *Note: the decorator sits between `@app.get` and the function. FastAPI's
    parameter introspection handles this fine — verified in practice, not just
    assumed.*

---

## 9. The demo argument

### Do not claim "it isn't probabilities"

It **is** probabilities. Claiming otherwise loses the argument to anyone who
reads one paragraph afterward.

### The defensible claim, in three separable parts

> **The randomness is optional, external to the model, and applied after the
> computation is finished. The computation itself is deterministic and highly
> structured.**

### Suggested running order

1. **T = 2.0 first** — show the gibberish. *"This is what a system that's
   actually guessing produces."*
2. **T = 0** — five runs, identical output. *"Same model, dice removed."*
   (Anticipate: "a lookup table is deterministic too." Fair — determinism kills
   *random*, not *unsophisticated*. Move to step 3.)
3. **T = 1.0** — point at the errors. Ginza is a *district of Tokyo*. Ryukyu
   are *Japanese islands*. Shizuoka is a *Japanese prefecture*. *"It's wrong.
   But it's wrong inside Japan, every time."* Guessing produces errors
   distributed at random; this produces errors clustered in the correct
   semantic neighborhood.
4. **The numbers** — 36.22% against a 0.00203% baseline. 17,803×. Ten tokens
   holding 69.8% of the mass.
5. **The number line** — throw 100 darts, watch observed frequencies converge
   on the segment widths. The mechanism, visible.
6. **Seed 42** — *"and even the randomness is reproducible."*
7. **The server log** — drag the temperature slider, nothing appears. *"The
   model already finished. This knob doesn't touch it."*
8. **A failure case** — non-negotiable. A demo that only shows wins gets
   discounted, correctly.

### Framing for the model's weak output

*"We're showing you the engine on a stand, not the car."* Same architecture,
same mechanisms, a tiny fraction of the scale.

---

## 10. Open questions and unverified items

- **Position convention in the J-lens.** Same-position slice here; Anthropic's
  description mentions current *and future* target positions. Their
  implementation was not read.
- **Whether the J-lens's layer-24 arrival is real insight or fitting artifact.**
  A linear map fitted to predict `h_final` will look prescient by design. No
  method identified to separate these.
- **12 prompts is a small fitting corpus.** Stability under a different corpus
  is untested.
- **Early-layer punctuation readouts.** "The answer isn't formed yet" and "the
  lens can't read this layer" look identical in the table. Evidence it is
  partly artifact: the two lenses disagree on *which* junk they report.
- **Whether any attention head shows clean Japan→France rerouting.** Untested.
  Middle layers, individual heads, is where to look.
- **The attention sink explanation.** The softmax-must-sum-to-1 account is
  widely repeated but the mechanism is still debated in the literature. Stated
  in the UI as a leading explanation, not a settled one.
- **Total installed disk footprint** of torch + transformers — never measured.

---

## 11. Suggested verification experiments

- **Temperature independence:** set the slider to 0.50 and re-run the lenses.
  Section 2's bars change; the lens table does not move.
- **Same seed, different prompts:** identical `r` sequence, completely
  different tokens.
- **Off-corpus prompt for the J-lens:** structurally unlike anything in the
  12-prompt fitting corpus. If the advantage collapses, it was fitting to
  prompt type.
- **J-lens on a prompt where the model is wrong** — distinguishes "reads the
  model's state" from "predicts the right answer."
- **`tokenizer.decode(range(273, 300))`** — look directly at the earliest BPE
  merges.
- **Rank lookup at 5,000 and 40,000** — see how far the ordering extends.
- **20 more T=1.0 samples** to test whether 16/20 in-domain holds.

---

## 12. Quick reference — the numbers

| Quantity | Value |
|---|---|
| Parameters | 134,515,008 |
| Layers / hidden / heads / KV heads | 30 / 576 / 9 / 3 |
| Vocabulary | 49,152 |
| Uniform baseline | 0.00203% |
| fp32 weights | 538.1 MB |
| Forward pass (6 tokens) | 61.1 ms |
| Jacobian cache | 41.1 MB, 22.8 min to fit |
| Special tokens / byte tokens / merges start | 17 / 256 / id 273 |
| `\|h\|` growth, embeddings → layer 29 | 2.5 → 729.1 (292×) |
| J-lens stage-30 diagnostic | 2.289e-05 PASS |

---

## 13. Remaining work

Documentation and demo preparation. No code is blocking.

1. **Attention exploration guidance panel** *(requested, deferred
   deliberately)* — a UI panel explaining where to look in the 270 grids, what
   distinguishes a real signal from a sink or a diagonal, how to run the
   Japan→France comparison, and what a null result would mean. Deferred until
   someone has actually explored the grids, so the guidance describes what's
   there rather than what's expected.
2. **Find a deliberate failure case** for the demo.
3. **Rehearse the talk track** in §9, especially the attention-sink
   explanation — it's the first thing anyone will point at.
4. Optional: position selector for the lens table, so any token's column can be
   inspected rather than only the last.
