# Mechanistic Interpretability Milestones

## 1. Artifact Safety and Reproducibility

Status: implemented

- Specify the supported Python version and optional PyTorch dependency.
- Pin the CPU-only PyTorch environment used for live analysis.
- Stream a SHA-256 digest for the model artifact.
- Record the artifact size, source URL, and expected checksum in JSON reports.
- Stop before pickle execution when a trusted checksum does not match.
- Test the static inspector with generated, non-executable PyTorch-style fixtures.

The Python 3.11+ artifact is present locally. Its streamed local SHA-256 exactly matches the digest
published by Hugging Face; see `ARTIFACTS.md`.

## 2. Architecture Recovery

Status: metadata exporter implemented; pending model artifact and hardened execution environment

- Identify the Python classes required to deserialize the model.
- Load the artifact inside a disposable environment with no network access and minimal
  filesystem access.
- Export the complete module tree, parameter and buffer shapes, and parameter counts. (implemented)
- Identify the final two computational layers and extract their weights and biases.
- Record input preprocessing and tokenization behavior.

## 3. Behavioral Probing

Status: pending

- Build a deterministic probe runner that stores inputs, outputs, model hash, and run metadata.
- Test controlled contrasts for order, capitalization, punctuation, repetition, and length.
- Compare semantic categories such as animals, foods, colors, places, and verbs.
- Use factorial experiments to separate lexical, positional, and interaction effects.

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
