# Transformer Learning Lab — version 2.1 update notes

Version 2.1 is primarily a learner-interface and documentation release. The
model calculations in `engine.py` are unchanged from version 2.0 apart from the
application version label.

## Changes

- Added an always-available glossary defining the AI and mathematical terms used
  throughout the lab, including tokens, logits, attention, Q/K/V, RoPE, GQA,
  residual stream, RMSNorm, MLP, inference, training, probes, and divergences.
- Expanded the seed lesson to explain pseudorandom generation, replay conditions,
  the `base_seed + run_index` rule, divergence after different sampled tokens,
  blank-seed behavior, and why temperature zero ignores the seed.
- Renamed **Predict next token** to **Analyze next-token distribution**. That
  action computes one distribution but does not sample and append a token;
  autoregressive continuation remains in **Generate and trace**.
- Added learner-oriented views for numeric arrays:
  - one-dimensional vectors receive a signed coordinate profile and histogram;
  - two-dimensional matrices receive a diverging heatmap;
  - long displays are grouped for rendering efficiency while preserving extrema
    in vector profiles and showing group means in matrix cells; and
  - exact unrounded values remain available in nested disclosures.
- Added definitions beside vector shape and L2 norm displays, plus warnings that
  coordinate adjacency, magnitude, and visual prominence do not establish
  semantic meaning, causal importance, or model knowledge.
- Added graphical views to the hand-checkable toy attention result, projection
  components, real transformer vectors and matrices, and the local Jacobian
  experiment.
- Clarified that the toy attention example omits RoPE by design and uses invented
  values for arithmetic that can be checked by hand.
- Added repository-level MIT licensing, release notes, and detailed development
  acknowledgments.

## Verification record

During preparation, 52 offline mathematical and JavaScript rendering tests
passed. That environment did not contain PyTorch; the limitations and exact
commands are preserved in `verification_preparation.json`.

After placing the complete v2.1 files in the Git working branch, the maintainer
ran the following checks on Windows on **2026-09-08**:

| Check | Result |
| --- | ---: |
| `python -m unittest discover -s tests -v` | 46/46 passed |
| `python verify_model.py --output <fresh-path>` | 201/201 passed |
| `node --test tests/test_math.js tests/test_ui.js` | 26/26 passed |

The first command includes 26 NumPy math tests and 20 model/API tests. The model
tests use a small random, untrained Llama fixture to test behavior. The second
command loads the actual pretrained SmolLM2-135M checkpoint. The JavaScript
tests use a DOM stub and are not full-browser end-to-end tests.

The Python checks emitted dependency deprecation warnings for a Starlette/AnyIO
alias and Transformers' legacy tuple-form KV cache. They were not failures. A
future dependency upgrade should migrate the cache interface and rerun the full
suite rather than hiding the warning.

These results verify the implemented checks on the tested environment. They do
not establish that every model interpretation is correct or that behavior is
identical under different prompts, model revisions, hardware, numerical kernels,
or software versions.
