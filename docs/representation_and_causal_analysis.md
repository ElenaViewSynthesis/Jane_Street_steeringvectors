# Representation and Causal Analysis

## Status

Milestone 4's three requested workstreams are implemented and exercised with validated reports.
The outcome is a bounded encoding conclusion, a rigorous semantic null result, and exact causal
agreement between recovered downstream algebra and interventions.

## Capture Boundary

The activation worker resolves and verifies the hash-bound final module sequence before installing
hooks:

```text
5437 ReLU             -> h192
5438 Linear(192, 48)  -> a48
5439 ReLU             -> z48
5440 Linear(48, 1)    -> readout preactivation
5441 ReLU             -> scalar output
```

Hooks are installed only around the complete inference loop and removed in the context manager's
exit path. Each tensor is required to be a finite CPU float32 tensor with its expected shape. The
report stores its values and a SHA-256 digest over canonical little-endian float32 encodings.

The worker independently verifies:

```text
a48 = W5438 h192 + b5438
z48 = ReLU(a48)
readout = W5440 z48 + b5440
output = ReLU(readout)
```

The trusted host accepts a report only when all reconstruction errors are at most `1e-4` (to allow
bounded float32 accumulation from additive directions), the returned scalar exactly matches the
final hook, the manifest and artifact hashes match, repeated activation digests agree with the
determinism flag, and the complete report passes its versioned schema.

## Smoke Experiment

`experiments/activations/m4-capture-smoke-v1.json` contains four inputs: the empty string, `abc`,
the documented `vegetable dog` example, and `red fox`. The suite was run twice per input with
Python 3.11.14, CPU-only PyTorch 2.7.1, and cloudpickle 3.1.1 inside the namespace and Landlock
sandbox.

All eight observations were bit-deterministic. Every affine and ReLU reconstruction error was
exactly zero. The individual target-match histogram was:

```text
0 matching target bytes: 6 observations
1 matching target byte:  2 observations
```

The one-byte match belongs to `abc` and is insufficient to pass the final 16-predicate gate.

## MD5 Hypothesis

For every smoke input, the 16 decoded predicate values exactly match the ordinary MD5 digest of
the unpadded ASCII input. None matches the MD5 digest of the model's 55-position null-padded input
representation.

This directly confirms that the captured final representation carries ordinary MD5 digest bytes
for the tested ASCII inputs. It also explains why broad scalar probing returned only zeros: the
published scalar checks the digest against the fixed target
`c7ef65233c40aa32c2b9ace37595fa7c` and discards all partial information.

### Boundary encoding experiment

The 10-case `m4-md5-boundary-v1` suite was captured twice per case. It establishes a 55 Python
character cutoff: 55 and 56 copies of `a` produce the same decoded digest. It also falsifies a
universal simple byte rule: `café` and `ÿ` agree with Latin-1/code-point byte candidates, whereas
the null, decomposed Unicode, BMP, and emoji cases agree with none of full/truncated UTF-8,
Latin-1, code-point-byte, or padded-code-point candidates. The exact MD5 byte encoding remains
unresolved outside the demonstrated short-ASCII and byte-range behavior.

## Held-out Semantic Directions

`m3-semantic-factorial-v1` was captured twice per case (450 observations). Directions are
normalized one-vs-rest `h192` mean differences, fit separately for left and right word slots.
Each of three folds holds out one lexeme from every category and excludes every pair containing a
held-out lexeme from training. The result is a null finding:

| Slot | Held-out accuracy | Chance |
|---|---:|---:|
| left | 0.180 | 0.200 |
| right | 0.207 | 0.200 |

Each fold also has 64 randomized-label controls. The observed accuracies are indistinguishable
from those controls, so this pipeline does not claim semantic directions. The analysis keeps an
explicitly leaky lexical-identity diagnostic separate from the held-out metric.

## Causal Downstream Tests

The intervention sandbox binds a float32-vector specification to the model, manifest, and source
activation-report hashes. A forward hook changes `h192`, then captures the effective `h192`,
`a48`, `z48`, readout preactivation, and final output. It reports predicted and observed deltas,
all predicate-ReLU crossings, and final-ReLU crossing status.

The full canonical control suite replaces `h192` with the recovered target bit layout on all four
smoke inputs: it produces 16/16 matches, readout `1.0`, and scalar `1.0`. Each of 16 controls
then flips one recovered predicate bit; each produces 15/16 matches, readout `0.0`, and scalar
`0.0`. The suite contains 136 intervention observations and all reported deltas agree exactly.

The complete semantic sweep executed all 30 held-out directions at strengths `-1`, `0`, and `1`
on their held-out category cases (1,800 intervention observations). Maximum errors were `0.0`
for predicate deltas, `3.0517578125e-05` for readout deltas, and `0.0` for output deltas. It
recorded predicate-ReLU crossings in 530 observations, no final-ReLU crossings, no nonzero scalar
outputs, readouts from `-15.000030517578125` to `-13.238693237304688`, and zero to one matched
predicate. This supports the recovered local piecewise-affine circuit while providing no evidence
that the directions are semantic.

## Reproduction

Use a Python 3.11 environment on the native Linux filesystem. On WSL, a virtual environment under
`/mnt/c` can be unreadable after the Landlock policy is applied.

```bash
python3 scripts/run_activation_sandbox.py \
  --venv /path/to/python-3.11-analysis-venv \
  --manifest experiments/activations/m4-capture-smoke-v1.json \
  --report outputs/activations/m4-capture-smoke-v1.json \
  --repetitions 2
```

Generated activation reports are ignored by Git. Their inputs and the aggregate findings are
committed in the manifest and this document.

## Reproduction of Causal Reports

```bash
python3 scripts/generate_intervention_specs.py --kind semantic \
  --manifest experiments/probes/m3-semantic-factorial-v1.json \
  --source-activation-report outputs/activations/m4-semantic-factorial-v1.json \
  --semantic-analysis outputs/activations/m4-semantic-direction-analysis-v1.json \
  --output outputs/interventions/m4-semantic-direction-interventions-v1.json

python3 scripts/run_intervention_sandbox.py --venv /path/to/python-3.11-analysis-venv \
  --manifest experiments/probes/m3-semantic-factorial-v1.json \
  --source-activation-report outputs/activations/m4-semantic-factorial-v1.json \
  --spec outputs/interventions/m4-semantic-direction-interventions-v1.json \
  --report outputs/interventions/m4-semantic-direction-report-v1.json
```
