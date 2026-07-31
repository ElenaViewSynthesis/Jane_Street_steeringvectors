# Representation and Causal Analysis

## Status

Milestone 4 is in progress. Deterministic final-representation capture and exact readout
verification are implemented. Held-out semantic-direction analysis and activation interventions
remain pending.

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

The trusted host accepts a report only when all reconstruction errors are at most `1e-5`, the
returned scalar exactly matches the final hook, the manifest and artifact hashes match, repeated
activation digests agree with the determinism flag, and the complete report passes its versioned
schema.

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

The current evidence does not yet determine behavior for non-ASCII inputs or every truncation
boundary. Those cases should be added before claiming a general text-to-byte encoding rule.

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

## Remaining Milestone 4 Work

- Capture non-ASCII and 55-character boundary cases to define the exact MD5 byte input.
- Extract candidate category directions with a lexeme-level train/held-out split.
- Evaluate direction projections against held-out labels and randomized-label controls.
- Add hash-bound interventions at `h192`, measuring predicate values, readout preactivation, and
  final output.
- Replace `h192` with a canonical target representation, then break one predicate at a time to
  verify the final gate causally.
