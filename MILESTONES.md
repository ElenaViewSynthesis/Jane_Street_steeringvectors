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

Status: implemented

- Capture intermediate activations with scoped hooks. (implemented for the final five modules)
- Relate activation directions to changes in the scalar output. (implemented with 1,800 bounded
  semantic-direction intervention observations)
- Test whether the final readout is linear in the last hidden representation. (verified exactly
  for the readout preactivation; the published scalar includes a final ReLU)
- Extract candidate semantic directions and validate them on held-out inputs. (implemented with
  whole-lexeme folds; left 0.180 and right 0.207 accuracy versus 0.200 chance)
- Perform activation interventions to distinguish causal features from correlations. (implemented
  with canonical gate controls and hash-bound additive/replacement interventions)

The recovered downstream algebra predicts every predicate delta exactly and every readout delta to
within `3.0517578125e-05`. Canonical target replacement produces output `1.0`; each single-predicate
break produces `0.0`. See `docs/representation_and_causal_analysis.md`.

## 5. Reproducible Findings

Status: implemented

- Maintain a hypothesis and falsification log. (`docs/research_log.md`)
- Save machine-readable experiment results under `outputs/`. (versioned local-geometry, inventory,
  and synthesis JSON reports; generated artifacts remain ignored by Git)
- Produce a deterministic analysis notebook or report. (`scripts/analyze_local_geometry.py` and
  `scripts/synthesize_findings.py` provide the canonical headless path)
- Document the inferred mechanism, supporting evidence, and remaining uncertainty.
  (`docs/final_report.md`, generated from validated source reports)

The local-geometry report builds the exact `16 x 192` predicate Jacobian, verifies it with
`torch.func`, emits direction and response Gram matrices, computes the 30-direction spectrum, and
reproduces all 530 observed predicate-ReLU crossing observations with zero mismatches. The
direction matrix has numerical rank 24. A six-block model of all 150 cross-fold cosine cells gives
a same-category-minus-control contrast of 0.309 (coherent 10,000-repetition permutation
p=0.0001), interpreted only as fitted-direction stability because held-out accuracy remains at
chance. Readout and output Hessians are exactly zero at a clean fixed-region point, so boundary
crossings and Jacobian jumps remain the primary nonlinear measurements. PyHessian is intentionally
not used because no parameter-space loss objective is defined.

The synthesis validates the architecture, probe, activation, intervention, semantic-analysis,
geometry, and plot hashes before writing its machine-readable summary, artifact inventory, and
human report. A fresh environment can regenerate the outputs using the committed manifests and
the documented Python 3.11 environments.
