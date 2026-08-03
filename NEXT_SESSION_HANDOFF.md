# Next Session Handoff

Last updated: 2026-08-03 (Europe/London)

## Resume Here

Start the next terminal in the repository root and read this file before changing code:

```bash
cd /mnt/c/Users/proxi/Documents/codex-jane/Jane_Street_steeringvectors
git status -sb
git log --oneline --decorate -8
python3 -m unittest discover -s tests -v
```

The repository is on branch `codex/local-work`. Before this handoff synchronization, the working
tree was clean at `608900b`, exactly matching `origin/main`. Milestones 1–5 are published and
complete; Milestone 6, the natural-text solution to the original puzzle, is pending. Always inspect
`git status -sb` before adding new work.

The latest published commits are:

```text
608900b docs(readme): publish local geometry visualizations
01dc091 docs(readme): add puzzle and Jacobian verification guides
0b8c53f docs(roadmap): define original puzzle completion
4654810 chore(security): exclude private project notes
8d2622e docs(analysis): publish Milestone 5 evidence and handoff
519b794 feat(skill): codify ReLU local geometry workflow
dd4def2 feat(reporting): synthesize validated milestone findings
92c98d7 feat(geometry): analyze recovered ReLU local structure
a595f7b build(analysis): isolate local geometry dependencies
57e2918 docs(project): add next-session implementation handoff
```

## Safety Boundary

`model_3_11.pt` is untrusted serialized input. Never call `torch.load` directly from an ordinary
process. Static inspection is safe through `scripts/inspect_model.py`; live loading and inference
must use the namespace and Landlock launchers.

Model provenance:

```text
file: model_3_11.pt
size: 1,158,729,818 bytes
SHA-256: 43aa7da7ccf749ae1fb95f8b7a6aa49536b73e27f0ac74cb90d5f824ccd484b2
runtime: exact Python 3.11
```

The working sandbox environment is on the native Linux filesystem:

```text
/home/proxi/.venvs/jsmi-py311
Python 3.11.14
torch 2.7.1+cpu
cloudpickle 3.1.1
```

Always pass it explicitly:

```bash
--venv /home/proxi/.venvs/jsmi-py311
```

Do not use a virtual environment under `/mnt/c` for sandboxed model execution. Landlock can deny
runtime reads through the WSL-mounted filesystem even when the environment works outside the
sandbox. NumPy is intentionally absent from the current sandbox environment; PyTorch may emit a
non-fatal warning about that.

The host's default `python3` is Python 3.12.3. It is suitable for the standard-library unit tests,
but not for loading the Python 3.11 cloudpickled artifact.

## What Is Implemented

### Milestone 1: Artifact safety

- streamed SHA-256 verification and provenance reporting;
- static ZIP and pickle-opcode inspection without payload execution;
- trusted artifact digest enforcement; and
- non-executable PyTorch-shaped test fixtures.

### Milestone 2: Architecture recovery

- isolated user, mount, network, PID, IPC, and UTS namespaces;
- Landlock filesystem allow-list, dropped capabilities, `no_new_privs`, and resource limits;
- full module and parameter metadata export; and
- recovery of the input wrapper and final predicate circuit.

The model is a `torch.nn.Sequential` circuit with 2,721 `Linear -> ReLU` stages and 288,998,553
float32 parameters. Its input wrapper converts `str(x)[:55].ljust(55, "\0")` to Unicode code
points and then to a 55-value float32 tensor.

### Milestone 3: Behavioral probing

- deterministic, manifest-bound scalar probe runner;
- 32-case controlled smoke suite; and
- complete ordered 15-by-15 semantic word-pair factorial.

All 514 scalar observations were deterministic zeros. The scalar alone is therefore unsuitable
for estimating lexical, positional, or interaction effects.

### Milestone 4: Representation and causal analysis

- scoped activation hooks for modules `5437` through `5441`;
- capture of `h192`, `a48`, `z48`, readout preactivation, and final output;
- algebraic verification of every affine and ReLU stage;
- MD5 candidate decoding and boundary manifests;
- whole-lexeme-held-out semantic direction analysis;
- hash-bound additive and replacement interventions at `h192`; and
- analytic-versus-observed predicate, readout, output, and ReLU-crossing reports.

Milestone 4 is complete and the project status documents are synchronized.

### Milestone 5: Reproducible findings

- exact direction, predicate-Jacobian, Gram, spectrum, boundary, and Jacobian-jump analysis;
- isolated PyTorch/SciPy/Matplotlib/Seaborn geometry requirements;
- deterministic generated-artifact inventory and findings synthesis;
- generated human-readable final report;
- hypothesis and falsification research log; and
- project-local `analyze-relu-local-geometry` skill.

The 30 directions have numerical rank 24 and entropy effective rank approximately 19.656. The
exact `16 x 192` predicate Jacobian has 192 nonzero entries and zero error against `torch.func`.
The six-block model of all 150 cross-fold cosine cells estimates a 0.309 same-category alignment
contrast (same-category mean 0.247 versus control mean -0.062; coherent 10,000-repetition
permutation p=0.0001). This is fitted-direction stability, while held-out semantic accuracy
remains at chance.
Of 600 nonzero direction/case endpoints, 265 cross a predicate ReLU. These reproduce all 530
repeated crossing observations with zero mismatches. Readout and output Hessians are exactly zero
at a clean fixed-region baseline.

PyHessian and BackPACK are not Milestone 5 dependencies. The analysis differentiates with respect
to the 192-dimensional `h192` activation and defines no parameter-space loss, labeled dataset, or
parameter subset. `torch.func` directly handles the exact activation derivatives. Introduce
parameter-curvature packages only after a future experiment versions those missing choices.

### Milestone 6: Original puzzle completion

Status: pending

The project has recovered the mechanism but has **not solved the original input challenge**.
`vegetable dog` is the official baseline and correctly returns `0.0`; its digest does not match the
target. Completion requires finding the intended two-lowercase-English-word input separated by one
space, then confirming 16/16 predicate matches, readout `1.0`, and output `1.0` in at least two
deterministic sandbox repetitions.

## Recovered Final Mechanism

`h192` is the output of module `5437` and the input to `Linear(192, 48)` at module `5438`. Its 192
coordinates form 24 little-endian bit-weighted blocks:

\[
q_k = \sum_{b=0}^{7} 2^b h_{8k+b}
\]

Those blocks form 16 predicate expressions. The next ReLU layer implements three-ReLU equality
indicators, and the final linear layer sums them with bias `-15`. The final scalar is positive only
when all 16 predicates match.

The fixed target digest is:

```text
c7ef65233c40aa32c2b9ace37595fa7c
```

For the tested short ASCII path, the 16 predicate bytes exactly equal ordinary MD5 of the unpadded
input. A positive short-ASCII input is therefore an MD5 preimage of this target.

Canonical causal controls verify the gate:

- canonical target `h192`: 16/16 matches, readout `1.0`, output `1.0`;
- each of 16 single-predicate breaks: 15/16 matches, readout `0.0`, output `0.0`; and
- 136 canonical intervention observations validated with exact predicted deltas.

## Milestone 4 Results

### Encoding boundary

The ten-case boundary suite was run twice per input and was deterministic.

- Inputs of 55 and 56 copies of `a` have the same decoded digest, confirming the 55-Python-character
  cutoff.
- A trailing null after `abc` is identical to the existing padding state and therefore decodes like
  `abc`.
- An embedded null followed by additional characters changes the digest.
- `café` and `ÿ` match one-byte Latin-1/code-point MD5 candidates.
- decomposed `café`, `漢`, and `😀` match none of the currently tested full/truncated UTF-8,
  Latin-1, raw code-point-byte, or padded-code-point-byte candidates.
- `漢` and `😀` produced the same decoded predicate digest in the current suite, suggesting an
  unsupported-code-point class or circuit collision that needs targeted testing.

Do not claim a universal UTF-8, Latin-1, modulo-256, or padded-byte encoding. The exact conversion
for code points above 255 remains unresolved.

### Semantic directions

The semantic-factorial activation report contains 450 deterministic observations. The current
estimator is a normalized one-vs-rest mean difference in `h192`, fitted separately for left and
right slots across three whole-lexeme holdout folds.

```text
left held-out accuracy:  0.180
right held-out accuracy: 0.207
chance accuracy:         0.200
```

This is a rigorous null result, not evidence of semantic structure. Each fold also includes 64
randomized-label controls.

The 30 directions are:

```text
2 slots × 3 folds × 5 categories = 30 directions
```

The semantic causal sweep is:

```text
30 directions × 10 held-out cases × 3 strengths × 2 repetitions = 1,800 observations
```

Results:

- strengths: `-1`, `0`, `+1`;
- exact maximum predicate-delta error: `0.0`;
- maximum readout-delta error: `3.0517578125e-05`;
- maximum output-delta error: `0.0`;
- predicate-ReLU crossings: 530 observations;
- final-ReLU crossings: 0;
- nonzero scalar outputs: 0; and
- observed readout range: approximately `-15.00003` to `-13.23869`.

## Local Generated Artifacts

All generated JSON reports are ignored by Git. They are present on this machine but will not exist
in a fresh clone. Preserve or reproduce them before relying on their paths. Three deterministic
PNG visualizations used by `README.md` are selectively tracked and will exist in a fresh clone;
the singular-value plot remains generated and ignored.

| Artifact | SHA-256 | Approximate size |
|---|---|---:|
| `outputs/reports/architecture_report.json` | `a9885c4a448fbccdaab5192b847162781418b7dd72ed2754201aa318919f518a` | platform dependent |
| `outputs/activations/m4-capture-smoke-v1.json` | `7cd2913cee17cbfaabf207603eaaed44b96283d2d864989a4bab513a7496d2da` | 96 KiB |
| `outputs/activations/m4-md5-boundary-v1.json` | `b2687f9da378a55d3c2e2f41e1ef6a10fd0478c773fb3640bf2dbf5d3933a0cb` | 228 KiB |
| `outputs/activations/m4-md5-boundary-analysis-v1.json` | `150b72ed86942168719738a9b70bab14ccf665a5e7b073510fea4c3f9def07a9` | 8 KiB |
| `outputs/activations/m4-semantic-factorial-v1.json` | `08bbc63423fb5d9e3d70ec3e54eeafcfe8ca01adfb8eb0d7baf343dbc3c73b93` | 5.0 MiB |
| `outputs/activations/m4-semantic-direction-analysis-v1.json` | `d5d299f6fc2b0b4ac8cf58757d80e9a5d4f110ee7d9f66e1f38a67b1f5ebb6ec` | 268 KiB |
| `outputs/interventions/m4-canonical-gate-v1.json` | `8eaae32752f74f9794141e05631f027f536ed49c21d19a5a8874dfed48130b37` | 52 KiB |
| `outputs/interventions/m4-canonical-gate-report-v1.json` | `6d0a041956d640c9c8f888bad555d30e8073333bdbc273c07b255b6ddaf15706` | 2.2 MiB |
| `outputs/interventions/m4-semantic-direction-interventions-v1.json` | `8326eaad91905e1766329887f4d4eee75a95805d11f01a5ff76cac2d33fd2664` | 184 KiB |
| `outputs/interventions/m4-semantic-direction-report-v1.json` | `8c611d0c90ecf01a82c3da74954652ac51b3ed9f6d656a9f6b08f8a15b439d94` | 38 MiB |
| `outputs/reports/m5-local-geometry-v1.json` | `8b95aac444e3ce930cb0f010060b434d2c01d95214a528810f40fa6b274c7d3c` | 1.4 MiB |
| `outputs/reports/m5-artifact-inventory-v1.json` | `a4ab3d16281098bfd101c3a68e0b09e474e1ea270f42ea34fbd939e13128cad9` | 4.4 KiB |
| `outputs/reports/m5-findings-summary-v1.json` | `5cf66aa95e1b1dba97266bc8a3abcd00bd1064f779cbec3d55caa2de205c06ed` | 5.9 KiB |

Tracked visualization assets:

| Artifact | SHA-256 | Size |
|---|---|---:|
| `outputs/reports/plots/m5-direction-gram-v1.png` | `dfa3f8e97f2de87748cc54798c669097012fa897a84ccc8cfe09537f4ccb7d98` | 103,297 bytes |
| `outputs/reports/plots/m5-response-gram-v1.png` | `5ffd8bc7f570c08ca1e6d2dccf7c11ddfdcece6861339fa05a176fb5aeaec1dc` | 104,200 bytes |
| `outputs/reports/plots/m5-cross-fold-cosine-model-v1.png` | `52d448eed924c5479df0e0fd2bd15ea3c20bfd5dccb38d6d93922f2529de52bf` | 227,727 bytes |

`outputs/activations/m4-capture-smoke-v1-new.json` is an ignored duplicate/temporary output and is
not part of the canonical artifact list above. Confirm its contents before deleting it.

## Reproduction Commands

### Unit tests

```bash
python3 -m unittest discover -s tests -v
```

Current result: 77 tests passing.

### Activation smoke

```bash
python3 scripts/run_activation_sandbox.py \
  --venv /home/proxi/.venvs/jsmi-py311 \
  --manifest experiments/activations/m4-capture-smoke-v1.json \
  --report outputs/activations/m4-capture-smoke-v1.json \
  --repetitions 2
```

### Encoding boundary capture and analysis

```bash
python3 scripts/run_activation_sandbox.py \
  --venv /home/proxi/.venvs/jsmi-py311 \
  --manifest experiments/activations/m4-md5-boundary-v1.json \
  --report outputs/activations/m4-md5-boundary-v1.json \
  --repetitions 2

python3 scripts/analyze_activation_report.py \
  --kind encoding \
  --manifest experiments/activations/m4-md5-boundary-v1.json \
  --report outputs/activations/m4-md5-boundary-v1.json \
  --output outputs/activations/m4-md5-boundary-analysis-v1.json
```

### Semantic activation capture and direction analysis

```bash
python3 scripts/run_activation_sandbox.py \
  --venv /home/proxi/.venvs/jsmi-py311 \
  --manifest experiments/probes/m3-semantic-factorial-v1.json \
  --report outputs/activations/m4-semantic-factorial-v1.json \
  --repetitions 2 \
  --output-mib 64

python3 scripts/analyze_activation_report.py \
  --kind semantic \
  --manifest experiments/probes/m3-semantic-factorial-v1.json \
  --report outputs/activations/m4-semantic-factorial-v1.json \
  --output outputs/activations/m4-semantic-direction-analysis-v1.json
```

### Canonical gate controls

```bash
python3 scripts/generate_intervention_specs.py \
  --kind canonical \
  --manifest experiments/activations/m4-capture-smoke-v1.json \
  --source-activation-report outputs/activations/m4-capture-smoke-v1.json \
  --output outputs/interventions/m4-canonical-gate-v1.json

python3 scripts/run_intervention_sandbox.py \
  --venv /home/proxi/.venvs/jsmi-py311 \
  --manifest experiments/activations/m4-capture-smoke-v1.json \
  --source-activation-report outputs/activations/m4-capture-smoke-v1.json \
  --spec outputs/interventions/m4-canonical-gate-v1.json \
  --report outputs/interventions/m4-canonical-gate-report-v1.json \
  --repetitions 2
```

### Semantic intervention sweep

```bash
python3 scripts/generate_intervention_specs.py \
  --kind semantic \
  --manifest experiments/probes/m3-semantic-factorial-v1.json \
  --source-activation-report outputs/activations/m4-semantic-factorial-v1.json \
  --semantic-analysis outputs/activations/m4-semantic-direction-analysis-v1.json \
  --output outputs/interventions/m4-semantic-direction-interventions-v1.json

python3 scripts/run_intervention_sandbox.py \
  --venv /home/proxi/.venvs/jsmi-py311 \
  --manifest experiments/probes/m3-semantic-factorial-v1.json \
  --source-activation-report outputs/activations/m4-semantic-factorial-v1.json \
  --spec outputs/interventions/m4-semantic-direction-interventions-v1.json \
  --report outputs/interventions/m4-semantic-direction-report-v1.json \
  --repetitions 2 \
  --output-mib 64 \
  --cpu-seconds 1800
```

The full semantic intervention run can take several minutes. When invoking it through an
interactive execution tool, start it as a persistent PTY session and poll rather than imposing a
30-second command timeout.

### Local geometry and Milestone 5 synthesis

Use a separate Python 3.11 environment installed from `requirements/geometry-analysis.txt`, then
run:

```bash
python3 scripts/analyze_local_geometry.py \
  --semantic-analysis outputs/activations/m4-semantic-direction-analysis-v1.json \
  --activation-report outputs/activations/m4-semantic-factorial-v1.json \
  --manifest experiments/probes/m3-semantic-factorial-v1.json \
  --intervention-spec outputs/interventions/m4-semantic-direction-interventions-v1.json \
  --intervention-report outputs/interventions/m4-semantic-direction-report-v1.json \
  --output outputs/reports/m5-local-geometry-v1.json \
  --plot-directory outputs/reports/plots

python3 scripts/synthesize_findings.py
```

The local temporary package target used during implementation was `/tmp/jsmi-geometry-packages`;
it is not part of the repository or the canonical environment. PyHessian and BackPACK were not
installed because the project defines no parameter-space loss objective.

## Recommended Next Work

**Highest priority:** complete the structured two-word target-preimage search described in item 2.
That is the only remaining task required to solve the original puzzle. The encoding, alternative
probe, and test-hardening items are useful research extensions but are not blockers for the
short-ASCII positive-input solution.

### 1. Resolve the code-point-above-255 encoding behavior

Create a second encoding manifest designed to localize thresholds and equivalence classes rather
than mixing unrelated examples. Suggested cases:

- code points `0`, `1`, `127`, `128`, `255`, `256`, `257`, `511`, `512`, `1023`, `65535`, and
  selected non-BMP values;
- the same code point at the first, middle, and 55th positions;
- isolated unsupported code points versus the same values surrounded by ASCII;
- embedded nulls followed by content at several positions; and
- pairs sharing low 8, low 16, or higher code-point bits.

Add candidate transforms only when they are explicitly named and falsifiable: modulo 256,
saturation, low-byte/high-byte decomposition, UTF-16 code units, UTF-32 endianness, and a shared
unsupported-character sentinel. Do not weaken report validation to make a candidate fit.

Acceptance criterion: either one deterministic transform explains all targeted cases or the
remaining equivalence classes and counterexamples are documented precisely.

### 2. Search for the target preimage in a structured candidate domain

For short ASCII, model evaluation is unnecessary during search because the circuit has been shown
to expose ordinary MD5. Build a safe, deterministic offline search script using `hashlib.md5`:

```text
target = c7ef65233c40aa32c2b9ace37595fa7c
```

Search plausible puzzle domains rather than attempting an infeasible unrestricted MD5 preimage:

- word pairs from curated category vocabularies;
- common English word lists and phrase lists;
- order, capitalization, whitespace, punctuation, and separators;
- singular/plural and simple morphological variants; and
- candidates suggested by the puzzle prompt and example `vegetable dog`.

Record the word-list provenance, candidate-generation grammar, count, and target digest. Verify
only a discovered hit through the model sandbox.

Acceptance criterion: the search space is reproducible and hash-only; either it finds a candidate
that the sandbox confirms with output `1.0`, or it records exactly which finite domain was
exhausted.

### 3. Add regularized-probe and uncertainty comparisons

Retain the mean-difference estimator as the baseline. Add optional comparisons using:

- one-vs-rest ridge coefficients mapped back into raw `h192` coordinates;
- shrinkage LDA as a lower-dimensional discriminant comparison;
- the exact existing whole-lexeme folds;
- lexeme-level label permutations; and
- lexeme-cluster bootstrap intervals.

Do not use random pair-level splitting. Paired inputs contain both left and right lexemes, so the
current rule excluding every training case containing any held-out word must remain authoritative.

Acceptance criterion: every estimator is evaluated on identical leakage-free folds and compared
with a grouped null distribution. A null result remains valid.

### 4. Strengthen intervention tests

The current tests cover vector hashing, manifest case binding, canonical target construction, hook
ordering, and sandbox command construction. Add focused tests for:

- complete intervention-report validation;
- report tampering at model/spec/source-report hashes;
- analytic comparison tampering;
- missing or duplicate intervention coverage;
- symbolic-link and output-size rejection during report publication; and
- canonical and semantic spec-generator counts and hashes.

Mirror the stronger publication tests already present for probe and activation reports.

## Important Code Map

| Path | Responsibility |
|---|---|
| `scripts/inspect_model.py` | Static archive inspection, trusted hashes, no direct pickle loading |
| `scripts/run_architecture_sandbox.py` | No-inference architecture recovery launcher |
| `scripts/run_probe_sandbox.py` | Scalar behavioral-probe launcher |
| `scripts/run_activation_sandbox.py` | Final-circuit activation capture launcher |
| `scripts/activation_worker.py` | Module resolution, scoped hooks, activation verification |
| `scripts/activation_schema.py` | Tensor/report schemas, predicate decoding, MD5 candidates |
| `scripts/analyze_activation_report.py` | Encoding summaries and held-out semantic directions |
| `scripts/generate_intervention_specs.py` | Canonical and semantic intervention specifications |
| `scripts/run_intervention_sandbox.py` | Hash-bound causal intervention launcher and publication |
| `scripts/intervention_worker.py` | Add/replace `h192`, run inference, capture causal results |
| `scripts/intervention_schema.py` | Spec/report validation and analytic downstream predictions |
| `scripts/analyze_local_geometry.py` | Exact direction/Jacobian geometry, crossings, Hessian checks, and plots |
| `scripts/local_geometry_schema.py` | Dependency-free geometry validation and exact predicate algebra |
| `scripts/synthesize_findings.py` | Validate canonical evidence and generate Milestone 5 reports |
| `docs/architecture_findings.md` | Input wrapper, architecture, predicate equations |
| `docs/representation_and_causal_analysis.md` | Milestone 4 evidence and reproduction |
| `docs/semantic_directions_and_local_geometry.md` | Gram/Jacobian/Hessian and probe-extension design |
| `docs/final_report.md` | Generated human-readable canonical findings |
| `docs/research_log.md` | Hypothesis, falsification, null, and open-question log |

## Repository and Tooling Notes

- Generated JSON reports and `model*.pt` are ignored intentionally. Do not use `git add -f` on
  them. The three README visualization PNGs are deliberate tracked exceptions.
- The current unit suite does not require PyTorch or the 1.16 GB model.
- Verify `gh auth status` before publishing. It succeeded for the latest direct-main pushes.
- Direct pushes requested by the user used `git push origin HEAD:main` from `codex/local-work`.
- Before any future push, fetch `origin/main` and confirm the update is a fast-forward.
- `.private/` is intentionally ignored. Never stage, quote, or publish its contents.
- Preserve detailed conventional commit bodies and split commits by analysis, sandbox,
  experiments, tests, and documentation when changes span those concerns.

## Suggested Fresh-Terminal Prompt

```text
Read AGENTS.md and NEXT_SESSION_HANDOFF.md completely. Inspect git status and current milestone
documents. Milestones 1–5 are complete, but the original puzzle is not solved. Continue with the
Milestone 6 two-word MD5 preimage search, preserving the sandbox boundary and validating any hit
through two deterministic model repetitions before documenting completion.
```
