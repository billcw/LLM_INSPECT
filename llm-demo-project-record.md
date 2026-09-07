# LLM Next-Token Demo — Complete Project Record

**Purpose:** a local, inspectable demonstration of how a language model predicts
the next token, built to show skeptical coworkers that the process is
structured computation rather than guesswork.

**Machine:** HP EliteBook, Windows 11, CPU only. No GPU used at any point.

**Status at time of writing:** working end to end. Sections 1–5 of the UI are
built and verified. Remaining work listed in §12.

---

## 1. Environment

Verified versions from the running system, not assumed:

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

**Run the server:**

```
uvicorn main:app --reload --port 8000
```

Then open `http://127.0.0.1:8000/`

### Environment notes learned the hard way

- **`Activate.ps1` is a PowerShell script.** In cmd.exe it silently does
  nothing — no error, no `(llmdemo)` prefix. Use `activate.bat` in cmd.
- **Python 3.14 is very new.** PyTorch supports it on Windows, but CUDA wheels
  do not yet exist for 3.14 — only CPU builds. For this project that is a
  convenience, not a limitation.
- **`--reload` restarts on file changes only.** Ctrl-C kills the server
  permanently. Use a second cmd window for other commands.
- **`torch_dtype` is deprecated** in transformers 5.16.1. Use `dtype=`.

---

## 2. The model: SmolLM2-135M

Configuration read directly from the loaded model, not from documentation:

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

Configs for v1 and v2 look nearly identical and get confused constantly. The
two fields that distinguish them, confirmed on this machine:

| Field | v1 | **v2 (this model)** |
|---|---|---|
| `rope_theta` | 10000 | **100000** |
| `max_position_embeddings` | 2048 | **8192** |

Layers, hidden size, vocab, and tied embeddings are identical between versions,
so those cannot be used to tell them apart.

### Why this model was chosen

Originally GPT-2 small was proposed, because it is the reference model for
nearly every interpretability tutorial — meaning its internals are checkable
against published work. SmolLM2-135M was chosen instead for better output
quality at similar size. The trade-off accepted: less external literature to
verify against, and a Llama-style architecture where the logit lens was known
to be less reliable. **That trade-off turned out to matter** — see §9.

### Architecture differences from GPT-2 that affected the build

1. **30 layers, 576 dims** (not 12 × 768). Taller lens table.
2. **Tied embeddings** — the unembedding matrix *is* the embedding matrix.
3. **RMSNorm, not LayerNorm.** Final norm is at `model.model.norm`, not `ln_f`.
4. **RoPE, not learned position embeddings.** Position is injected by rotation
   inside attention, so the embedding row of the residual stream is *pure token
   identity*, position-free.
5. **GQA: 9 query heads, 3 KV heads.** Attention weights are per query head.

---

## 3. Core concepts

### 3.1 What the model does

One thing only: given text, produce a probability for every one of the 49,152
tokens in its vocabulary, describing how likely each is to come next. It does
not plan a sentence, does not look anything up, and does not have memory
between runs.

**The guessing baseline:** `1 / 49,152 = 0.00203%`. Every probability observed
should be compared against this number.

### 3.2 Tokens, not words

Text is split into tokens — chunks from a fixed vocabulary built by BPE
(byte-pair encoding). Common words are usually one token; rare words are split.
Most tokens carry a **leading space**: `" capital"` is a different token from
`"capital"`.

Consequence: when `" Os"` or `" Sh"` appears in predictions, the model is
starting to spell *Osaka* or *Shizuoka* and will finish on the next pass. It
predicts tokens, not words.

### 3.3 The grid: positions × layers

A transformer does not process a sentence as one object. **Every token gets its
own 576-number vector**, and those vectors travel through the 30 layers *side by
side, in parallel*.

So there is no single "vector for the prompt." There is a grid:

- Columns = token positions (set by the tokenizer)
- Rows = 31 stages (embeddings + 30 layer outputs)

For a 6-token prompt: 6 × 31 = **186 separate 576-number vectors**.

The next-token prediction is read from exactly one cell: **top row, rightmost
column**. Every other cell feeds into it indirectly through attention.

This is why the UI needs both a layer selector and a position selector — the
grid is the natural coordinate system.

### 3.4 The residual stream — vectors are added to, not replaced

**This was the single biggest conceptual correction in the project.**

A common assumption: the embedding assigns *the* vector for a token, and layers
replace it with new ones. Neither is right.

Each layer does this:

```
h = h + attention(norm(h))
h = h + mlp(norm(h))
```

Note the `h +` on both lines. Each sublayer computes a **correction** and adds
it to what is already there. It does not overwrite. 30 layers × 2 sublayers =
**60 additions** stacked on the original embedding.

```
h_final = embedding + (60 corrections summed)
```

The embedding vector is still present at layer 30 — mathematically there, never
deleted. The layers did not replace the token's meaning; they annotated it with
context. This is called the **residual stream**, and the skip connection is the
architecture's central trick.

**Measured evidence:** the vector magnitude `|h|` grows from **2.5** at the
embedding stage to **729.1** at layer 29 — roughly **292×** growth, which is the
60 corrections accumulating as a measurable quantity.

(The apparent drop to 44.2 at stage 30 is *not* a contradiction: HuggingFace's
`hidden_states[-1]` has already had the final RMSNorm applied, which rescales
it. See §6.2.)

### 3.5 Causal masking

Every position can attend to itself and to positions *before* it, never after.

Consequence, tested directly:

- **Appending** tokens to a prompt leaves earlier positions' activations
  unchanged: max abs difference **1.526e-05**
- **Prepending** tokens changes them completely: max abs difference **1.245e+01**

Six orders of magnitude apart. The append difference is floating-point rounding
noise, not a real change — BLAS libraries can choose different summation orders
for different matrix shapes, and floating-point addition is not associative.

**This is why KV caching works**, and why generation gets cheaper per token
rather than more expensive.

### 3.6 The forward pass

One complete trip through the model: tokens in, 49,152 logits out. "Forward"
means data flows one direction only — no loops, no revision. (The *backward*
pass only happens during training. The weights here are frozen.)

Two properties that surprise people:

1. **Nothing is stored between passes.** The model has no memory of previous
   runs. It "remembers" earlier text only because that text is fed in again as
   input.
2. **Computation is fixed.** An easy prompt and a hard prompt cost exactly the
   same. The model cannot think longer about a harder question.

Measured: **61.1 ms** for a 6-token prompt on this machine.

### 3.7 Unembedding — 576 numbers become 49,152 scores

At the end of a forward pass you hold 576 numbers and need 49,152 scores. The
bridge is a matrix of shape **49,152 × 576**.

**Each row of that matrix is a 576-number vector representing one vocabulary
token.** The multiplication is 49,152 separate dot products:

```
logit for " Tokyo" = dot( h_final , row_for_Tokyo )
```

A dot product measures alignment. The final step asks, 49,152 times: *how much
does the vector I built point the same direction as this token's vector?*

**Tied embeddings:** `tie_word_embeddings = True` means the unembedding matrix
**is** the embedding matrix — the same table used in both directions.

- Read one row → token becomes a vector (start of the pass)
- Dot against all rows → vector becomes token scores (end of the pass)

So the whole forward pass is: look a token up in the table, spend 30 layers
adding corrections to it, then ask the same table which token the result most
resembles.

**Empirical confirmation:** in the lens table, the embeddings row reads
`called 100.0%` when the last input token is `" called"`. With tied embeddings
that vector *is* that token's row in the unembedding matrix, so dotting it
against all rows returns itself at maximum.

### 3.8 Logits, softmax, and temperature

**Logits** are the raw output: 49,152 unbounded scores. Not probabilities — they
can be negative and don't sum to anything.

**Softmax** converts them: exponentiate each, divide by the total.

**Temperature** divides the logits first:

```
p = softmax(logits / T)
```

- Small T spreads logits apart → softmax concentrates
- Large T squashes them together → distribution flattens
- T = 0 collapses onto the single highest logit (greedy)

**Critical:** the logits never change when temperature changes. The model has
already finished. Temperature only reshapes how its finished answer is read.

Useful identity: `p_T ∝ p_1^(1/T)`, so a temperature-adjusted distribution can
be computed from a T=1 distribution without re-running the model.

### 3.9 How a token is actually selected — the number line

The "raffle ticket" intuition is essentially correct. The actual algorithm:

1. Softmax the logits → 49,152 probabilities summing to 1
2. **Cumulative sum** them → boundaries on a number line from 0 to 1, each token
   occupying a segment as wide as its probability
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
would need a million tickets. On the number line it just gets a segment
0.000001 wide. Arbitrary precision, no counting.

**What temperature does here:** it resizes the segments before the dart is
thrown. Same tokens, same order, different widths. At T = 0.5 the top segment
above swells from 27.8% to 73.4% of the line.

**At T = 0 there is no dart at all.** `argmax` takes the widest segment; the
PRNG is never called.

### 3.10 The random number knows nothing

The PRNG has never seen the prompt, the logits, the tokens, or the model. In
code it is one line:

```python
next_id = int(torch.multinomial(sample_probs, num_samples=1))
```

That line would work identically on a weather model or a hand-typed list of
five probabilities. It does not care what the numbers mean.

**But the number line is not independent.** The model does all the work of
deciding *what the segments are*. The random number does all the work of
deciding *where the dart lands*. The dart is blind; the board is not.

Precise phrasing that matters: `r = 0.15` is not a choice about tokens. It
becomes `know` only because the model put `know` at 0.0–0.278. Change the
prompt and the same `r` lands on something entirely different.

**Seeds:** fixing the seed fixes the sequence of `r` values, making sampled
output exactly reproducible. Even the randomness is a deterministic algorithm
producing a fixed sequence from a starting value.

### 3.11 Autoregression

The model cannot write a sentence. It predicts one token. Generation is a loop:

1. Tokenize the prompt
2. Run one forward pass → 49,152 logits
3. Pick one token
4. **Append that token to the input**
5. Return to step 2 with the now-longer input

Generating 12 tokens = 12 complete forward passes through all 30 layers.
Nothing is planned ahead; each token is chosen knowing only what came before.

**This explains many failure modes.** Once a wrong token is appended it is in
the context permanently. The model cannot go back and revise it.

---

## 4. The two lenses

### 4.1 Logit lens

Normally the model converts to vocabulary scores once, at the end. The logit
lens asks: what if we did it early?

```
readout(h_L) = lm_head(norm(h_L))
```

Take the residual stream after layer 6, apply the same final normalization and
the same unembedding matrix, see what token it points at. Do this at all 31
stages for a trace of the prediction assembling.

**Its weakness:** it pretends layers L+1 through 30 do not exist. It reads the
stream as if computation stopped there.

**Its deeper assumption:** the unembedding matrix was trained to read the
*final* layer. Applying it to layer 6 assumes the residual stream keeps a
consistent reading frame throughout. This holds reasonably for some models and
poorly for others.

### 4.2 Jacobian lens

Anthropic's J-lens (released July 2026, paper: *Verbalizable Representations
Form a Global Workspace in Language Models*) adapts the logit lens. Instead of
pretending the remaining layers don't exist, it approximates them with a single
averaged linear map:

```
readout(h_L) = lm_head(norm(J_L · h_L))
J_L = E[ ∂h_final / ∂h_L ]      averaged over a corpus
```

In words: *if I nudge this number in the stream at layer L, how does the final
vector move, on average?* One matrix multiply stands in for everything layers
L+1 through 30 would have done.

**Cost difference:** the unembedding matrix already exists in the weights, so
the logit lens is free. The Jacobians do not exist — they must be computed by
running the model with automatic differentiation over a corpus, and cached.

**Caveat on this implementation:** Anthropic's expectation is described as
running over current *and future* target positions. This implementation takes
the same-position slice (∂h_final[last] / ∂h_L[last]). It is a J-lens in the
same spirit, **not a reproduction of theirs**. Their implementation was not
read.

### 4.3 How the Jacobians were computed

The technique is a **forward hook**:

1. Run the model normally, capture the residual stream at layer L
2. Make a copy marked `requires_grad=True`
3. Install a hook that swaps the copy in at layer L
4. Run again — now `h_final` connects to the copy through the autograd graph
5. Each backward pass yields **one row** of the 576 × 576 Jacobian

**Important subtlety:** `model.requires_grad_(False)` is set, which looks like
it should break this. It does not. We differentiate with respect to an
*injected activation*, not with respect to the weights. Parameter gradients and
activation gradients are separate things; disabling parameter gradients just
makes the graph smaller and the fit faster.

**Timing measured on this machine:**

| Path | per row | one Jacobian | 30 layers × 10 prompts |
|---|---|---|---|
| Sequential backward | 24.4 ms | 14.0 s | 1.2 hours |
| Batched (`is_grads_batched=True`, vmap) | 4.5 ms | 2.6 s | ~13 min |

Batched path is **5.4× faster**. Actual fit: **22.8 minutes** for 12 prompts,
producing a **41.1 MB** cache.

### 4.4 The free correctness check

At stage 30 the Jacobian is the identity matrix by definition — there is no
remaining computation to approximate. Therefore the J-lens **must** reproduce
the model's real logits exactly there.

Verified results:

- Stage 30 Jacobian vs identity: **0.000e+00**
- Frobenius norm: **24.00** (identity of dim 576 = √576 = 24.0)
- J-lens output vs real logits at stage 30: **2.289e-05, PASS**

If that number were not near zero, the entire J-lens column would be
meaningless.

---

## 5. Files built

| File | Purpose |
|---|---|
| `probe.py` | Standalone diagnostic: environment, config, tokenization, top-k, causal-mask test |
| `main.py` | FastAPI server: `/api/predict`, `/api/generate`, `/api/lenses`, `/api/health`, static mount |
| `static/index.html` | Single-page UI with inline explanations |
| `jacobian_probe.py` | Timing probe — measures Jacobian cost before committing to a fit |
| `fit_jacobian.py` | Two-phase: verify in seconds, then fit and cache 30 Jacobians |
| `jacobian_cache.pt` | 41.1 MB cached Jacobians (12-prompt corpus) |
| `token_probe.py` | Tokenizer inspection — how strings split, and token IDs |

**Endpoint costs, for reference:**

| Endpoint | Forward passes | Note |
|---|---|---|
| `/api/predict` | 1 | ~61 ms |
| `/api/generate` (n=12, runs=5) | 60 | ~3.7 s |
| `/api/lenses` | 1 | but 61 unembedding readouts ≈ 1.73 billion MACs — exceeds the forward pass itself |

---

## 6. Empirical findings

### 6.1 Prediction quality

**"The capital of Japan is"**

| token | prob | × uniform |
|---|---|---|
| ` Tokyo` | 18.28% | 8,985× |
| ` the` | 13.07% | 6,424× |
| ` located` | 6.14% | |
| ` Kyoto` | 3.31% | 1,627× |

**"The capital of Japan is called"** — adding one word nearly doubled confidence:

| token | prob | × uniform |
|---|---|---|
| ` Tokyo` | **36.22%** | **17,803×** |
| ` Kyoto` | 6.32% | |
| ` N` | 6.01% | |
| ` As` | 5.93% | |
| ` the` | 3.78% | (dropped from rank 2 to rank 5) |

Top 10 tokens (0.02% of the vocabulary) hold **69.8%** of total probability.

**"The capital of France is"** — weaker: `Paris` came **second** at 9.38%,
behind `the` at 26.17%. Demo recommendation was to lead with Japan.

### 6.2 The 20-run structured-failure audit

Twenty samples at T = 1.0, "The capital of Japan is called ___":

| category | count |
|---|---|
| `Tokyo` | 10 / 20 |
| Real Japanese place (Tokyo, Nara ×2, Kyoto ×2, Kyushu, Fushimi) | **16 / 20** |
| Fabricated but Japanese-phonetic (Japhethius, Agivao, Yamato-Komusubutsu) | 3 / 20 |
| Non-Japanese place | **0 / 20** |

Nara and Kyoto are *actual former capitals of Japan*.

**Necessary qualification:** the *answer slot* never leaves Japan across 20
runs. The *continuation text* does drift — "Pearl River Delta" (China) and
"Korean money" appear after the answer. Correct claim: *the direct answer stays
in Japan; the text after it degrades.*

### 6.3 Determinism

- T = 0, 5 runs: **all identical** — "The capital of Japan is Tokyo."
- T = 1.0, 10 runs: **10 distinct outputs**
- T = 1.0 with fixed seed: identical across runs

At T = 0.10, five runs produced identical text for **eleven consecutive
tokens**, then split at a paragraph boundary where several sentence-openers had
comparable probability. A clean illustration of the number line.

### 6.4 Lens comparison results

| | reaches top-5 | reaches #1 |
|---|---|---|
| **Logit lens** | stage 30 | stage 30 |
| **Jacobian lens** | **stage 24** | **stage 24** |

The logit lens **never finds the answer early at all.** From layer 1 through
layer 23 it sits between rank #8,000 and #14,000 while reporting commas and
periods at 90%+ confidence. It drops to #413 at layer 24 and reaches #1 only at
the final layer.

The J-lens reaches #1 at layer 24 and **holds it for seven consecutive layers**.
Layers 22–23 are visibly assembling the *concept* before the specific answer:

- Layer 22: `Hare 2.5% Constantinople 1.9% Capital 1.6%` → rank #47
- Layer 23: `Capital 4.9% City 4.8% Cities 4.4%` → rank #10
- Layer 24: `Tokyo 72.1% Japan 25.9%` → rank #1

**29 of 31 stages disagree** on the top token between the two lenses.

**Jacobian Frobenius norm by stage** (identity = 24.0), showing non-monotonic
structure with no confident interpretation:

```
stage  1: 24.97    stage 18: 25.88
stage  2: 21.48    stage 24: 35.59
stage  3: 19.57    stage 28: 31.41
stage  6: 17.19    stage 29: 30.00
stage 12: 21.72    stage 30: 24.00
```

### 6.5 The `ot` token investigation

Question: why does token `ot` appear in the top 10 after "I don"? Typos of
"donot"?

**Tokenizer probe results:**

```
'donot'       -> 2 tokens  ids=[12420, 320]     pieces=['don', 'ot']
'cannot'      -> 1 token   ids=[33551]
'pseudonotum' -> 4 tokens  ids=[47438, 258, 320, 382]
                            pieces=['pseud', 'on', 'ot', 'um']
```

**The decisive test:** `ot` is the *same token, id 320*, in both `donot` and
`pseudonotum`. The latter has nothing to do with "do not." So `ot` is a
general-purpose subword fragment, not a "donot" artifact.

**What id 320 means:** BPE builds merges in frequency order. With ~272 IDs
occupied by specials and base bytes, id 320 is roughly the **48th merge ever
made** — top 0.10% of merges. "ot" is one of the most common two-character
sequences in English (not, got, hot, lot, spot, foot, rot).

**The strongest counter-evidence to the typo hypothesis:** `cannot` is a single
token — BPE merged it, meaning it appeared often enough to earn a vocabulary
entry at 6 characters. `donot` never did. If "donot" typos were common enough
to matter, BPE would likely have merged it too.

**Confound to note:** ID order conflates frequency with *length*. A 6-character
token cannot merge until its pieces exist. Only same-length comparisons are
sound.

**Verdict:** `ot` is there because "ot" is an extremely common letter pair, not
because people misspell "do not." The typo hypothesis survives only as a
possible minor contributor to a 0.015% residue.

### 6.6 Temperature recovery from a screenshot

A displayed table showed `'t` at 66.01% while the summary line reported
"48748× more likely than chance" (= 99.18% raw). Those disagree because the
probability column is temperature-adjusted and the summary uses raw T=1 values.

Solving from the top-2 logit gap of 5.43: **T ≈ 1.75**.

Consequence: `ot` displayed at 0.4357% is actually **0.0149%** at T = 1 — about
**7× uniform**, essentially the noise floor of a distribution where the winner
takes 99.18%.

---

## 7. Errors made and corrected

Kept deliberately, because the corrections are part of the record.

1. **"Exactly identical" on the causal-mask test.** Predicted exact zero; got
   1.526e-05. Correct phrasing is "identical up to floating-point rounding."
   The underlying claim about causal masking held — the six-order-of-magnitude
   gap versus prepending is what confirms it.

2. **T = 0 top-k tie-break bug.** Ranking by temperature-adjusted probabilities
   at T = 0 produced a 49,151-way tie broken by vocabulary index, displaying
   `<gh_stars>`, `<reponame>` and similar as apparent predictions. **Fix:**
   always rank by raw logits; use adjusted values for display only.

3. **`torch_dtype` vs `dtype`.** Deprecated in transformers 5.16.1.

4. **Contradictory instructions** for where to place `app.mount` — said both
   "near the top" and "at the very end" in the same paragraph. Correct rule:
   the mount catches all unmatched paths, so it must come *after* the API route
   definitions.

5. **Sanity check placed after the expensive operation.** The first
   `fit_jacobian.py` ran 22.5 minutes and *then* reported failure. Rewritten as
   two phases: verify in seconds, abort on failure, fit only if passing.

6. **The `hidden_states` norm bug.** `hidden_states[30]` has already had the
   final RMSNorm applied, while `hidden_states[1..29]` have not. Injecting the
   normed tensor where the unnormed one belongs produced a stage-30 Jacobian
   that was the *RMSNorm* Jacobian (max deviation 2.116, Frobenius 22.88 vs
   24.0) rather than the identity. **Fix:** capture the pre-norm output of the
   last decoder layer by hook.

7. **Stale caveat about `torch.set_grad_enabled(False)`.** Raised before the
   fit-offline-and-cache design was chosen, then never retracted. It became
   moot: `fit_jacobian.py` never disables grad globally, and `main.py` never
   computes a Jacobian.

8. **Wrong prediction about lens convergence.** Predicted disagreements would
   "cluster in the middle layers and vanish near the end." Actual: 29 of 31
   rows disagree, persisting through layer 29 and vanishing only at stage 30
   where it is mathematically forced.

9. **Over-strong claim about structured failure.** Initially stated every T=1.0
   error stays inside the Japanese semantic neighborhood. True for the answer
   slot (0/20 non-Japanese), but the continuation text does drift out of Japan.

---

## 8. Known bugs still present

1. **Stale tokenization display.** The token chips only refresh when
   `runPrediction()` fires (Enter key or slider release). Typing a new prompt
   leaves the old tokens on screen — a real problem for live demos.

2. **Redundant forward passes on temperature change.** Every slider move
   triggers a full 30-layer forward pass producing byte-identical logits. In
   one observed session: 18 predict requests, 2 distinct prompts, **16
   redundant forward passes** (~0.98 s wasted).

   This inefficiency is *itself evidence* for the demo thesis: if temperature
   were part of the model, you would have to re-run the model to change it. You
   don't.

---

## 9. The demo argument

### Do not claim "it isn't probabilities"

It **is** probabilities. The final layer is literally a distribution over
49,152 tokens. Claiming otherwise loses the argument to anyone who reads one
paragraph afterward.

### The defensible claim, in three separable parts

> **The randomness is optional, external to the model, and applied after the
> computation is finished. The computation itself is deterministic and highly
> structured.**

### Suggested running order

1. **T = 2.0 first** — show the gibberish. *"This is what a system that's
   actually guessing produces."*
2. **T = 0** — five runs, identical output. *"Same model, dice removed."*
   (Anticipate: "a lookup table is deterministic too." That is fair —
   determinism kills *random*, not *unsophisticated*. Move to step 3.)
3. **T = 1.0** — point at the errors. Ginza is a *district of Tokyo*. Ryukyu are
   *Japanese islands*. Shizuoka is a *Japanese prefecture*. *"It's wrong. But
   it's wrong inside Japan, every time."* Guessing produces errors distributed
   at random; this produces errors clustered in the correct semantic
   neighborhood.
4. **The numbers** — 36.22% against a 0.00203% baseline. 17,803×. Ten tokens
   holding 69.8% of the mass.
5. **Seed 42** — *"and even the randomness is reproducible."*
6. **A failure case** — non-negotiable. A demo that only shows wins gets
   discounted, correctly. Showing a failure buys credibility for everything
   else.

### Framing for the model's weak output

*"We're showing you the engine on a stand, not the car."* Same architecture,
same mechanisms, a tiny fraction of the scale.

---

## 10. Open questions and unverified items

- **Position convention in the J-lens.** Implementation takes the same-position
  slice; Anthropic's description mentions current *and future* target
  positions. Their implementation was not read.
- **Whether the J-lens's layer-24 arrival is real insight or fitting artifact.**
  A linear map fitted to predict `h_final` will look prescient by design. No
  method identified to separate these.
- **12 prompts is a small fitting corpus.** Stability under a different corpus
  is untested.
- **Early-layer punctuation readouts.** "The answer isn't formed yet" and "the
  lens can't read this layer" look identical in the table. Evidence it is
  partly artifact: the two lenses disagree on *which* junk they report.
- **Merge-number offset.** The ~272 base offset used in §6.5 was inferred from
  the T=0 tie-break, not read from the tokenizer. Relative ordering is
  unaffected.
- **Total installed disk footprint** of torch + transformers on this machine —
  never measured.

---

## 11. Suggested verification experiments

- **Temperature independence:** set the slider to 0.50 and re-run the lenses.
  Section 2's bars change; the lens table does not move. Confirms the lens table
  never applies temperature.
- **Same seed, different prompts:** identical `r` sequence, completely different
  tokens. Demonstrates the PRNG's independence directly.
- **Off-corpus prompt for the J-lens:** a prompt structurally unlike anything in
  the 12-prompt fitting corpus. If the advantage collapses, it was fitting to
  prompt type.
- **J-lens on a prompt where the model is wrong.** Would distinguish "reads the
  model's state" from "predicts the right answer."
- **`tokenizer.decode(range(272, 300))`** — look directly at the earliest BPE
  merges.
- **20 more samples** at T = 1.0 to test whether the 16/20 in-domain figure
  holds.

---

## 12. Remaining work

1. **Fix the stale tokenization display** — refresh chips on typing.
2. **Cache logits in the browser** — recompute softmax in JavaScript on slider
   moves. Demo payoff: *"watch the server log while I change temperature"* and
   nothing appears.
3. **Visualize the number line** — horizontal bar with proportional segments and
   a marker where the dart landed. Would make §3.9 visible rather than
   described.
4. **Attention heatmaps** — the most defensible visual, token × token, with real
   axes. Not yet built.
5. **A deliberate failure case** for the demo.
6. Optionally: position selector for the lens table, so any token's column can
   be inspected, not just the last.
