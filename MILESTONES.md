# Mechanistic Interpretability Milestones

## 1. Artifact Safety and Reproducibility

Status: implemented

- Specify the supported Python version and optional PyTorch dependency.
- Pin the CPU-only PyTorch environment used for live analysis.
- Stream a SHA-256 digest for the model artifact.
- Record the artifact size, source URL, and expected checksum in JSON reports.
- Stop before pickle execution when a trusted checksum does not match.
- Test the static inspector with generated, non-executable PyTorch-style fixtures.

The Python 3.11 artifact is present locally. Its streamed local SHA-256 exactly matches the digest
published by Hugging Face; see `ARTIFACTS.md`.

## 2. Architecture Recovery

Status: implemented

- Identify the Python classes required to deserialize the model. (implemented)
- Load the artifact inside a disposable environment with no network access and minimal
  filesystem access. (implemented with namespaces and Landlock)
- Export the complete module tree, parameter and buffer shapes, and parameter counts. (implemented)
- Identify the final two computational layers and extract their weights and biases. (implemented)
- Record input preprocessing and tokenization behavior. (implemented)

The model is a 2,721-stage piecewise-linear circuit over 55 character code points. Its last two
parameterized layers implement 16 equality predicates followed by an AND-like threshold. See
`docs/architecture_findings.md`.

## 3. Behavioral Probing

Status: implemented

- Build a deterministic probe runner that stores inputs, outputs, model hash, and run metadata.
  (implemented with a hash-bound manifest and validated report schema)
- Test controlled contrasts for order, capitalization, punctuation, repetition, and length.
  (implemented with the 32-case smoke suite)
- Compare semantic categories such as animals, foods, colors, places, and verbs.
  (implemented with a balanced 15-word vocabulary)
- Use factorial experiments to separate lexical, positional, and interaction effects.
  (implemented as a complete ordered 15-by-15 word-pair design)

Both suites were executed twice per input inside the namespace and Landlock sandbox. All 514
observations were deterministic zeros: 64 observations from the smoke suite and 450 from the
semantic factorial. Because the scalar output has no variance over these probes, lexical,
positional, and interaction effects cannot be estimated from the final output alone. See
`docs/behavioral_probing.md`.

## 4. Representation and Causal Analysis

Status: pending

- Capture intermediate activations with scoped hooks.
- Relate activation directions to changes in the scalar output.
- Test whether the final readout is linear in the last hidden representation.
- Extract candidate semantic directions and validate them on held-out inputs.
- Perform activation interventions to distinguish causal features from correlations.

## 5. Reproducible Findings

Status: pending

- Maintain a hypothesis and falsification log.
- Save machine-readable experiment results under `outputs/`.
- Produce a deterministic analysis notebook or report.
- Document the inferred mechanism, supporting evidence, and remaining uncertainty.
