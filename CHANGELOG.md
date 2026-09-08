# Changelog

This file records user-facing changes. Experimental findings belong in the
historical project records or exported verification reports, not in this release
log.

## [2.1.0] — 2026-09-08

### Added

- Always-available glossary defining the AI and mathematical terms used in the
  lessons.
- Deeper random-seed explanation covering pseudorandom streams, replay
  conditions, `base_seed + run_index`, divergence after sampling, blank-seed
  behavior, and why greedy decoding at temperature zero ignores the seed.
- Signed coordinate profiles and histograms for vectors.
- Diverging heatmaps for matrices.
- Explicit visual caveats: coordinate adjacency, magnitude, and salience do not
  establish semantic meaning or causal importance.
- Repository-ready MIT license and expanded attribution.

### Changed

- Renamed **Predict next token** to **Analyze next-token distribution** because
  the action computes one distribution but does not sample and append a token.
- Kept exact numeric arrays accessible beneath graphical summaries.
- Clarified the hand-checkable toy attention explanation.
- Expanded installation, verification, privacy, scientific-boundary, and
  repository-structure documentation.

### Verified

Maintainer-run Windows checks on 2026-09-08:

- Python math, model, and API suite: 46/46 passed.
- Actual SmolLM2-135M checkpoint verifier: 201/201 passed.
- JavaScript math and DOM-stub rendering suite: 26/26 passed.

The Python run emitted dependency deprecation warnings for Starlette/AnyIO and
the legacy Transformers tuple-form KV cache. Neither warning failed a check.
The JavaScript suite is not a real-browser end-to-end test.

## [2.0.0] — 2026-09-07

Version 2 was a substantial reconstruction of the original teaching tool.

### Added or rebuilt

- Separate `engine.py`, `math_core.py`, and browser assets instead of a monolithic
  HTML application.
- Stable full-vocabulary softmax and sampling math with an explicit temperature
  zero branch, exact tail handling, and deterministic tie rules.
- Full transformer-block inspection: RMSNorm, Q/K/V, grouped-query mapping,
  RoPE, masking, attention-weighted values, output projection, residual writes,
  gated MLP, final normalization, and unembedding.
- Seeded multi-run generation, optional KV cache, EOS control, processed-position
  accounting, and step traces.
- Prompt comparison with full-distribution and layer-state metrics.
- Head-level attention views and controlled attention interventions.
- Shifted-target loss/perplexity, a separate analytic optimizer example, affine
  probes, and a local Jacobian-vector-product experiment.
- Model provenance, request bounds, serialized instrumentation, safe hook cleanup,
  tests, in-app verification, notes, and JSON export.

### Corrected or removed

- Replaced the unsupported averaged `jacobian_cache.pt` lens with experiments
  whose limitations and checks are explicit.
- Separated raw final hidden state from final RMSNorm rather than obscuring their
  relationship.
- Distinguished attention display normalization from an actual intervention.
- Removed claims that a high attention weight, a norm, or a probe result alone
  explains model behavior.

## [1.x] — 2026-09-06 and earlier

- Initial local browser-based SmolLM2 inspection tool.
- Tokenization, logits, temperature and sampling, generation traces, layer
  readouts, attention displays, and exploratory Jacobian scripts.
- Historical scripts and project records are retained under `archive/` and
  `docs/history/` for auditability; they are not part of the supported v2 runtime.
