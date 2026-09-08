# Acknowledgments and development provenance

LLM_INSPECT is an AI-assisted project directed, evaluated, and published by Bill
(`billcw`). This account is intentionally specific about the collaboration so
readers can assess both the project's strengths and its limitations.

## Human direction and verification

Bill:

- identified the central teaching problem and intended non-specialist audience;
- supplied prompts, questions, desired experiments, and interface feedback;
- ran the application and its test suites on the target Windows environment;
- tested the actual SmolLM2-135M checkpoint rather than relying only on toy data;
- challenged interpretations and required false or overstated claims to be
  corrected;
- chose the retained features, licensing approach, repository organization, and
  release path; and
- remains responsible for deciding what is published under his account.

The archived project records preserve parts of that iterative process, including
ideas that were revised or rejected.

## Anthropic Claude

The original implementation was created with substantial assistance from
Anthropic's Claude over multiple conversations. Most of the original Python and
JavaScript was produced in those sessions, together with explanations and early
diagnostic scripts. Those files and records provided the starting point for the
later reconstruction.

## OpenAI Codex

OpenAI Codex performed a later scientific and code audit and substantially
reconstructed version 2. Its contributions included separating the runtime into
testable modules, revising numerical and sampling behavior, expanding
transformer-block instrumentation, adding controlled interventions and
verification checks, replacing or qualifying unsupported interpretability
claims, strengthening API and local-use safeguards, writing tests and learning
documentation, and adding the version 2.1 glossary, seed lesson, and numeric
visualizations.

## How to interpret this credit

Claude and Codex are software systems, not independent witnesses or substitutes
for review. Agreement between AI systems is not validation. Generated code and
prose were treated as drafts subject to human direction, selection, execution,
testing, correction, and release decisions.

Acknowledgment does not imply that Anthropic or OpenAI sponsors, endorses, or
accepts responsibility for this project. It also does not change the licenses of
the SmolLM2 model or any dependency.

## Contributions

Corrections are welcome, especially when they include a reproducible prompt,
model and tokenizer identity, dependency versions, the exact measurement or
claim at issue, and a minimal test when practical. Interpretability claims should
distinguish descriptive evidence, interventions, causal conclusions, and
speculation.
