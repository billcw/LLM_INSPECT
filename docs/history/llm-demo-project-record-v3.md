# LLM Interpretability Demo — Project Record v3

**Supersedes:** `llm-demo-project-record-v2.md`
**Date of this revision:** 5 September 2026
**Status:** Tool complete and working. Attention exploration substantially done.
Two items from v2 §13 remain open; one new high-value item added.

---

## 0. How to use this document

v2 described a finished tool. This revision covers a single long working session
that changed the tool in five patches and produced findings that were not
anticipated when v2 was written.

**Read §1 if you are picking this up cold.** Read §5 before describing anything
in this project to another person — it lists claims that were made and later
disproven, several of them by Claude, and it is the section most likely to stop
a repeat.

The failure mode this document exists to prevent: in the previous session, a
correction was written out in a *different* chat thread and never applied to the
file, because the record did not capture it. It sat wrong in `index.html` for
weeks. Do not let this document fall behind the code again.

---

## 1. Environment and stack (unchanged from v2)

- Windows 11, HP EliteBook, **CPU only**
- `C:\Projects\llm-demo`, venv `llmdemo`, activated by `activate.bat` in `cmd.exe`
- Python 3.14.7, torch 2.13.0+**cpu**, transformers 5.16.1, fastapi 0.141.1, uvicorn 0.52.4
- Model: `HuggingFaceTB/SmolLM2-135M` — 134,515,008 params, 30 layers, hidden 576,
  vocab 49,152, 9 query heads / 3 KV heads, RMSNorm, RoPE, tied embeddings
- Backend `main.py` (FastAPI), frontend `static/index.html` (single page, no build step)
- Offline fitting: `jacobian_probe.py`, `fit_jacobian.py` → `jacobian_cache.pt`
- **TransformerLens deliberately not used**

`main.py` was **not modified** in this session. All changes are in `index.html`.

---

## 2. What changed in `index.html` this session

Five patches, applied in order. Each was gated on post-conditions that were
checked *before* the file was written; two patches aborted on a failed gate and
were corrected rather than forced. Line endings preserved as CRLF throughout.

Size progression: 51,245 → 70,462 characters (LF-normalised count), then
→ 73,205 on 6 September 2026 (patch 6), then → 75,413 (patch 7, same day).

### Patch 1 — the two stranded panels

The correction written in the *transformers* chat thread and never applied.

- **"Two things you'll see that are easy to misread"** rewritten. Now opens with
  the arithmetic floor (row *i* has *i*+1 unmasked cells, so uniform attention
  alone puts 1/(*i*+1) on column 0), attributes the softmax account as "usually
  attributed to" rather than "the leading explanation", and adds the
  StreamingLLM evidence.
- **"Why 9 heads"** rewritten. Heads are 64-dimensional *slices* (576 = 9 × 64),
  not nine copies of the computation. Adds the additive o_proj point, the GQA
  read-vs-write distinction, the KV-cache arithmetic, and an explicit statement
  that nobody has mapped these 270 heads.

### Patch 2 — stale panels, baseline scale, max aggregation

- **Stale-panel bug fixed.** `runPredict()` now clears sections 4, 5 and 6 when
  the prompt changes. Previously you could change the prompt, hit Predict, and
  leave the old lens table and attention grid on screen with no warning. This
  was the single most likely way to mislead someone during a live demo.
- **New Scale control:** `raw %` / `× uniform`. In ratio mode every cell is
  divided by 1/(*i*+1). 1.0× means indistinguishable from uniform.
- **New Head option:** `max of all 9` — per-cell maximum across heads.
- `escapeHtml` now escapes double quotes (a prompt containing `"` previously
  broke the tooltip markup).
- Hint under the attention controls rewritten; it previously pointed people at
  a Japan→France test that cannot work in the default view.

### Patch 3 — legibility and correctness of the ratio view

- Cells below **0.25×** render blank. Previously amber intensity was `0.5×(1−r)`,
  so the *least* informative cells were shaded hardest. In a long prompt near-zero
  is the default state by arithmetic, not a finding.
- Row 0 prints `1.0` instead of rendering blank and looking broken.
- "Rows sum to 1." was **false** next to the ratio grid; now view-aware.
- Column-0 summary no longer claims rising multiples mean a strengthening sink
  (see §5.1). It now reports the live mean and range of the raw share.
- Dart tally clamps the tail at zero — at low temperature float error printed
  `-0.00%` for a probability.

### Patch 4 — visible whitespace, hatched mask, max-view caveat

- **`tokenLabel()`** added. Whitespace-only tokens render as `\n`, `␣`, `\t`;
  empty as `∅`. Tokens with a *leading* space are left alone deliberately,
  because the `" capital"` vs `"capital"` distinction is a teaching point in §1
  of the page. Applied in six places.
- Masked cells hatched (45° stripe) so "causally impossible" and "below the
  display floor" no longer look identical.
- Max view carries its own selection-bias warning (see §4.3).

### Patch 5 — sink profile chart

New section under the attention grid. Computed **client-side from the already
cached attention** — one Load click fetches all 270 grids, so this costs nothing.

For every layer, takes the **bottom row** and plots three quantities as multiples
of that row's uniform baseline:

| series | meaning |
|---|---|
| sink | column 0 |
| self | the diagonal |
| best other (not sink/self) | largest weight on anything else |

Log y-axis, dashed red reference line at 1.0×, follows the Head selector,
vertical marker for the layer shown in the grid above, hover for exact values,
live caption naming peak/trough layers.

### Patch 5b — chart relabel

The third series was originally labelled **"content (best other)"**. That label
asserted meaning the number does not carry — see §5.4. Renamed to
**"best other (not sink/self)"**, with the explanation panel and caption
corrected to match.

---

### Patch 6 — the honest-framing panel (6 September 2026)

Ships §7. One new `<details class="explain caution">` in section 6, placed last,
immediately above the grid controls, so it is the final thing read before anyone
touches the grid. Summary: *"What this grid cannot tell you — and the control
built in for it."*

Six paragraphs: weights are half the operation (values are not shown); a bright
cell is a hypothesis, not a finding; Jain & Wallace and Serrano & Smith; the
Wiegreffe & Pinter uniform-weights baseline, named as what the `× uniform`
control already is; a **scope caveat** (all three papers studied single-attention-
layer text classifiers, not 30-layer decoders — the mechanical argument carries,
the measurements do not); and a pointer back to section 5 as where the meaning
lives.

No CSS, no JavaScript, no new classes. Gates passed before writing: anchor unique
(the first anchor tried matched twice and was widened), details 19/19, div count
unchanged, each of Jain / Wiegreffe / Serrano exactly once, CRLF preserved with
zero lone CR or LF, `node --check` clean on the script block.

---

### Patch 7 — scenario-proofing (6 September 2026)

Prompted by the question "are we patching for specific values, or any scenario?"
An audit found the page teaches mechanism and math (general) and lets the visitor
generate their own numbers (general code), with exactly two prompt-specific
residues in prose plus one reading trap. All three fixed; `main.py` audited and
needed nothing (all its outputs are computed per request; the only constants are
capacity limits).

- **7a** — the `|h|` "2.5 → over 700" sentence (a measurement stated as a model
  property, prompt unrecorded) became `#hnorm-note` with a hedged general
  fallback; `renderLenses()` now overwrites it with the live values from
  `stages[0]` and `stages[length−2]` (last pre-norm stage, layer-count-agnostic)
  of the visitor's own prompt.
- **7b** — the sink chart now draws the hard ceiling at n× (one cell = 100% of a
  1/n baseline), labelled with this prompt's n; the y-range always includes it;
  tooltips gained the raw percent, which *is* comparable across prompts; the
  caption states baseline, ceiling, and the comparison rule.
- **7c** — the chart panel's "values span roughly 0.1× to 25×" (true only of the
  26-token prompt it was written on; impossible on a 6-token prompt, whose
  ceiling is 6×) replaced with a general statement plus a paragraph on why
  multiples are length-dependent and heights must not be compared across prompts.

Gates: anchor uniqueness per edit, details/div counts unchanged, `<p>` +1
balanced, CRLF clean, `node --check` clean. First run aborted on a miscounted
`<p>` gate (+2 expected, +1 correct) — the gate was wrong, the edit was not;
nothing was written until the gate was fixed.

Remaining known prompt-specific content on the page: none found. The "on one
prompt / on another" phrasing in the chart panel and the six-token worked
example in §6's misread panel are conditioned in the text itself and stand.

---

## 3. Verified against the source paper

The StreamingLLM claims in the sink panel were checked directly against the
arXiv HTML this session, not taken from memory.

**Xiao, Tian, Chen, Han, Lewis — *Efficient Streaming Language Models with
Attention Sinks*, ICLR 2024.** arXiv 2309.17453.

- Perplexity figures are in **Table 1**, Section 3.1, under "LLMs attend to
  Initial Tokens as Attention Sinks". Llama-2-13B on the first book (65K tokens)
  of the PG-19 test set:
  - `0 + 1024` (window, no initial tokens): **5158.07**
  - `4 + 1020`: **5.40**
  - `4"\n" + 1020` (initial four replaced by line breaks): **5.60**
- **They coined the term.** The Introduction states they term these tokens
  "attention sinks". They named the phenomenon; they did not first observe the
  underlying oddity, which the paper traces to earlier quantization-outlier work.
  "Named" is the correct verb; "discovered" is not.
- Tested across **ten models**: Llama-2-[7,13,70]B, Falcon-[7,40]B,
  Pythia-[2.8,6.9,12]B, MPT-[7,30]B. (The Introduction writes Pythia 2.9B; §4.1
  writes 2.8B. 2.8B is correct — do not quote 2.9.)
- Appendix H reports the same phenomenon in **BERT**, on `[SEP]`.
- **Figure 2 / Appendix F:** in Llama-2-7B, layers 0 and 1 show a *local* pattern
  with recent tokens favoured; the sink appears beyond the bottom two layers.
  This became the testable prediction in §4.1 below.

Related but **not verified this session:** the representational-collapse /
over-mixing account of *why* models learn sinks. That clause exists in the UI
panel and traces to a different paper that was never opened. Flagged in §6.

---

## 4. Findings on SmolLM2-135M

All findings are from **layer-by-layer attention on two prompts** unless stated.
Prompt A: `The horse ate all of the oats, but none of the hay.  I'm taking a wild
guess that it doesn't like` (26 tokens, contains an accidental double space —
see §4.4). Prompt B: `Horse ate oats, but no hay. I'm taking a wild guess that
it doesn't like` (20 tokens, no determiners).

### 4.1 The layer profile — replicated across both prompts

| | Prompt A (25 tok, 3× `the`) | Prompt B (20 tok, no `the`) |
|---|---|---|
| weakest sink | layer 2, **0.60×** | layer 2, **0.27×** |
| crossover | ~12–13 | ~12–13 |
| sink peak | layer 27, **19.9×** | layer 27, **16.7×** |
| best-other peak | layer 5, 10.9× | layer 9, 14.3× |

**Layers 1–2 have no sink** — column 0 sits at or below uniform, and attention is
local/diagonal. This reproduces StreamingLLM's Figure 2 prediction on a model
50× smaller from a different training run.

Weakest at layer 2 and peak at layer 27 in **both** prompts. The profile is a
property of the model, not the sentence.

### 4.2 Head specialization is enormous

Layer 30, prompt "The capital of Japan is called", bottom row, column 0:

| head | raw | × uniform |
|---|---|---|
| 1 | 30.3% | 1.82× |
| 2 | 18.7% | 1.12× |
| 3 | 95.6% | 5.74× |
| 4 | 9.2% | **0.55×** — below uniform |
| 9 | 97.3% | 5.84× |
| mean of 9 | 65.4% | 3.92× |

Head 9 is a pure sink: every row 94–100% on position 0. Head 4 steers *away*
from it and reads recent tokens instead. **The mean of 65.4% describes no head
that exists.** Heads 5–8 were never checked; arithmetic implies they average
~87%.

### 4.3 The max view is biased upward — quantified

Taking the largest of 9 draws per cell inflates everything. Null simulations run
this session (unspecialized heads, realistic sink share, Dirichlet spreading):

- **max-of-9, single cell:** median 1.3×, p90 2.2×, p99 3.2× (50% sink share)
- **max-of-9, largest of 25 interior cells in a row:** median 2.8×, p95 3.8×,
  p99 4.4×; never reached 5.4× in 20,000 trials
- **mean-of-9, largest interior cell:** median 1.06×, p99 1.48×, max 1.93×

Working thresholds, now stated in the UI: in the max view treat under ~4× as
noise and 5×+ as real. In the mean view the null is much tighter, so 2×+ is
already signal.

Caveat: the null assumes independent unstructured heads, which understates real
head correlation. It is a calibration, not a significance test.

### 4.4 The double-space artifact

Bill types two spaces after a period. In prompt A the tokenizer gave one space
to `I` (position 15 = `" I"`) and the other became a **standalone token at
position 14** (`" "`). Confirmed by header tooltips, not inferred.

That invisible token then received **33.0% of the budget (8.6× uniform)** from
the final position in the max view at layer 30 — the second-largest target after
the sink and the diagonal, and beyond anything the null produced.

**Deleting the space does not relocate the behaviour.** With single spacing, the
period at that position gets **6.15% (1.54×)** from the same query token, which
is inside the noise band. The head was locked to that token at that position,
not to "the nearest structural marker".

No mechanism is offered for this. Present it without one.

### 4.5 Content routing is rare everywhere

Layer 30, prompt A, max of 9, bottom row:

- `hay` #12 — 20.9% = **5.4×** (beyond the null; real)
- `oats` #6 — 11.4% = 2.9× (noise band)
- `wild` #19 — 10.7% = 2.8× (noise band)

Layer 5, same prompt, **mean** of 9: `oats` 0.09×, `hay` 0.12× — actively
suppressed. Layer 5, **max** of 9: `oats` 0.37×, `hay` 0.71× — still below
uniform. **No head at layer 5 retrieves the nouns.**

Instead, the early-middle peak lands on function words:
- Prompt A, layer 5: `the` #5 at 43.7% mean / 53.5% max, plus `the` #11
- Prompt B, layer 9: **the comma** #4 at **71.5%**, and column 4 reads 60–77% in
  *every* row from index 4 to 19

Removing all determiners moved the anchor to a comma and made it *stronger*.
The pattern is punctuation and function words as positional anchors, not
determiners specifically.

**Summary claim, defensible across both prompts and all 30 layers:** attention
concentrates on position 0, on itself, and on the nearest punctuation or
function word. The nouns the sentence is about sit at or below uniform nearly
everywhere, with one 5.4× exception at layer 30 that required the max view to find.

### 4.6 Incidental

- In prompt B, `Horse` tokenizes as `H` + `orse` — capitalised word without a
  leading space is rare in training text. Free reinforcement of the §1 panel.
- Position 0 in prompt B is `Horse`, a content noun, and the sink still forms on
  it (16.7×). Stronger demo than a sink on `The` — but note `it` corefers with
  the horse, so someone may argue position 0 is semantically relevant here.
- A trailing space in the prompt makes the *query* token a space, which changes
  what the bottom row means. Watch for it.

---

## 5. Corrections made during this session

Each of these was asserted and then disproven. They are listed because the
disproofs cost real time and would cost it again.

### 5.1 "The sink holds a constant ~54% of the budget"

Claimed after seeing rows 6–13 of a 12-token prompt, where the raw share is
genuinely flat (sd 2.1). **Falsified** by a 26-token prompt: rows 14–25 have
sd 10.7 and range 38–82%. The ratio is not even monotone — it falls at 7 of 24
transitions. The *level* holds (~52% mean); the *stability* does not.

Correct statement: the sink holds roughly half the budget regardless of context
length, varying ±10 points row to row once the sentence gets long. The ratio
trends up because the baseline shrinks, not because the sink grows.

### 5.2 "Position 14 is a newline"

Inferred from prompt layout, then repeated with more confidence than the
evidence supported. **It is a space.** Settled by a header tooltip, after the
tooltip was built for exactly this purpose.

### 5.3 "3× is the noise threshold in the max view"

Too low. 3.2× is the *99th percentile* of pure noise for a single cell. Correct
thresholds in §4.3.

### 5.4 "Content lives at layer 12" / the chart's "content" label

The layer-12 dip is real, but it came from reading screenshots of a different
prompt using max-across-rows rather than the bottom row. The bottom-row chart
puts the best-other peak at layer 5 (prompt A) and layer 9 (prompt B), and in
both cases **the peak is a function word, not content**. The chart series was
relabelled in patch 5b.

### 5.5 "Attention shows what the model looked at"

The framing the whole demo implicitly rested on. See §7.

---

## 6. Open caveats carried in the UI

- The over-mixing / representational-collapse clause in the sink panel is
  sourced to a prior conversation, not to a paper opened in this project.
  Either verify it or cut it.
- Every attention finding is **one aggregation, two prompts**. Layers 4, 7–11,
  13–17, 19–23 and 25–29 have never been looked at.
- The J-lens performs much worse on off-corpus prompts. On "The cow ran to the "
  it disagreed with the logit lens at 28 of 31 stages and produced tokens like
  `firehose` and `coals` through the early layers, only becoming useful around
  layer 21. The stage-30 diagnostic still passed (3.481e-5). This is the
  12-prompt-corpus limitation the caution panel already warns about, visible in
  practice. **Do not lead with the J-lens on an off-corpus prompt.**
- Two panels contain numbers measured on the *default* prompt: the `" Os"` /
  `" Sh"` spelling example in the tokens panel, and "`|h|` grows from about 2.5
  to over 700" in the residual-stream panel. Both are hedged, neither is wrong,
  but don't read them aloud with a different prompt on screen.

---

## 7. The framing this session produced

Verified this session: **Jain & Wallace, *Attention is not Explanation*, NAACL
2019** (arXiv 1902.10186, ACL Anthology N19-1357). Attention weights are
frequently uncorrelated with gradient-based feature importance, and very
different attention distributions can yield equivalent predictions. Serrano &
Smith, *Is Attention Interpretable?* (**ACL** 2019, arXiv 1906.03731, anthology
P19-1282, pp. 2931–2951) zeroed attention weights one at a time and found the
highest weight predicts impact only noisily: it frequently fails to have a large
effect, and gradient-based rankings often predict effects better than the weights
themselves do. **An earlier draft of this section said they found zeroing the top
weight "often left the prediction unchanged" — that is stronger than the paper
and was corrected on 6 September 2026.** Wiegreffe & Pinter (EMNLP 2019,
anthology D19-1002, pp. 11–20) rebutted — attention is
not *the* explanation, but may be *an* explanation — and among their four
proposed tests is a **uniform-weights baseline**, which is what the `× uniform`
scale independently reimplements.

The mechanical reason no visualization fixes this: **attention weights are half
the operation.** A head weights positions, averages the *value* vectors, and
writes through its output slice into the residual stream. The grid shows the
weights and not the values, so a bright cell can write nothing and a dim cell
can write a lot.

**The honest line for the demo:** this view shows where each position spent its
attention budget. It does not show what was written into the residual stream, so
a bright cell is a hypothesis, not a finding. Here is the published result
saying so, and here is the uniform baseline the rebuttal recommends, which is
the control built into this tool.

Sections 5 and 6 of the page **can now be presented as** one argument: attention
shows scaffolding, and the meaning is in the residual stream.

**History of this sentence.** It originally read "are now one argument", which
reads as a claim that this framing had been written into `index.html`. It had
not — §7 recorded a conclusion, not a patch, and §2 lists no patch that shipped
it. Shipped on 6 September 2026 as patch 6. Wording corrected the same day.

---

## 8. Remaining work

### 8.1 Attention knockout endpoint — **highest value, new**

Zero a chosen head at a chosen layer via a forward hook, re-run the forward
pass, show the before/after prediction. Roughly forty lines of Python plus a
small frontend.

This converts the tool from observational to causal, which is exactly the move
the field made after Jain & Wallace. Two immediate experiments:
- Knock out head 9 at layer 30 (the pure sink head) and see whether the output
  degrades. StreamingLLM predicts it should matter a lot.
- Knock out a head with a bright content cell and see whether anything moves.

### 8.2 Attention guidance panel (v2 §13 item 1)

No longer blocked — the grids have now been explored extensively. Should be
written from §4 above.

### 8.3 Deliberate failure case (v2 §13 item 2)

Still open.

### 8.4 Talk track rehearsal (v2 §13 item 3)

Still open. Should now incorporate §7.

### 8.5 Model-agnostic port — **parked, scoped**

Target hardware: Minisforum UM760 + Razer Core with **RTX 3060 12GB**.

*Ports with no work:* `/api/tokenize`, `/api/decode`, `/api/predict`,
`/api/generate` — they only touch `AutoTokenizer` / `AutoModelForCausalLM`.

*Needs a resolver:* `/api/lenses` hardcodes `model.model.norm` and
`model.lm_head`. Llama-family naming. GPT-2 uses `transformer.ln_f`. Must fail
loudly, not silently read the wrong tensor.

*Needs rewriting:* the frontend is a hardcoded curriculum — `VOCAB = 49152`,
"30 layers", "576-dimensional", "576 = 9 × 64", the 45 KiB KV-cache figure,
"60 additions". This is the bulk of the work and it is writing, not engineering.

*Does not scale:* the Jacobian lens. Cache size goes as hidden² × layers —
40 MB here, ~470 MB at 1.7B, **2.1 GB at 8B**, 4.2 GB at 14B; refit time roughly
23 min → 4.5 h → 21 h → 40 h. The logit lens scales for free. Let the J-lens
report "not fitted" on other models, which the code already handles.

*VRAM on 12GB (weights only; the 32-token cap makes activations negligible —
2 MB attention, 9 MB hidden states at 8B):* bf16 comfortable to ~3B; 7–8B needs
8-bit; 14B needs 4-bit.

**Hard gate, unverified:** does a CUDA torch wheel exist for **Python 3.14 on
Windows**? Check with
`pip index versions torch --index-url https://download.pytorch.org/whl/cu124`
before anything else. If not, the answer is a second venv on 3.12, not a code
change.

**Use bf16, not fp16** (Ampere supports it natively; wider exponent range). Note
this breaks on-screen numbers measured at fp32 — the causal-mask deviation
1.526e-05 and the J-lens 1e-3 PASS threshold both need to scale with dtype, or a
*correct* J-lens will fail the gate.

**First port target: GPT-2 small**, not a big model. 12 layers, 768 hidden,
12 heads, 50,257 vocab, ~500 MB, runs on the current CPU setup. It breaks the
right things: module paths, LayerNorm instead of RMSNorm, learned positional
embeddings instead of RoPE, plain MHA instead of GQA. Side by side with SmolLM2
it shows that "transformer" is a family, not a thing — and that the sink appears
in both across a seven-year architecture gap.

### 8.6 Loose end

Heads 5–8 at layer 30 on the Japan prompt were never checked. Predicted to
average ~87% on column 0. Four clicks.

---

## 9. Working practices that paid off

- **Post-condition gates before writing files.** Two patches aborted on a failed
  gate this session; both were the gate catching a wrong prediction about the
  edit, not a wrong edit.
- **Null simulations before calling a number a finding.** Three claims were
  demoted to noise this way.
- **Hover tooltips carrying positions and raw JSON strings.** Built specifically
  because two whitespace tokens were indistinguishable on screen; immediately
  resolved a question that had been guessed wrong twice.
- **Running one more prompt.** Every correction in §5 came from this.
