# Architecture Recovery Findings

## Execution Boundary

The model was loaded with Python 3.11.14, CPU-only PyTorch 2.7.1, and cloudpickle 3.1.1 after its
SHA-256 was verified twice: once by the trusted host launcher and again inside the restricted
worker immediately before `torch.load`.

The worker ran with:

- separate user, mount, PID, network, IPC, and UTS namespaces;
- a network namespace with no external interfaces;
- a Landlock ABI 7 filesystem allow-list;
- read-only access to the staged model, analysis runtime, and trusted scripts;
- write access only to private temporary and report-staging directories;
- an empty environment apart from explicitly supplied runtime variables;
- all capabilities dropped and `no_new_privs` enabled; and
- CPU, address-space, output-size, file-descriptor, and process-count limits.

No model inference was executed during architecture recovery.

## Input Representation

The root object is a `torch.nn.Sequential`, but its instance `_call_impl` is replaced with a
cloudpickled lambda. Python 3.11 bytecode disassembly reconstructs it as:

```python
lambda x: model.forward(
    torch.Tensor(
        list(map(ord, str(x)[:55].ljust(55, "\x00")))
    )
)
```

Consequences:

- This model does not use a tokenizer or learned word embeddings at its input.
- The input is converted with `str`, truncated to 55 Python characters, and padded on the right
  with null characters.
- Each character becomes its Unicode code point through `ord`.
- `torch.Tensor` converts the resulting 55 numbers to a float32 vector.
- Text after character 55 cannot affect the output.

The cloudpickled code is Python-version-specific. Although Python 3.12 can deserialize the object,
its opcode interpretation is not valid Python 3.11 disassembly. The hardened launcher therefore
requires an exact Python 3.11 runtime for `model_3_11.pt`.

## Network Architecture

The loaded model has:

- root type: `torch.nn.Sequential`;
- 5,442 child modules;
- 2,721 `Linear` modules;
- 2,721 `ReLU` modules;
- strict `Linear -> ReLU` alternation, including the output layer;
- 288,998,553 trainable float32 parameters; and
- no registered buffers.

The network begins with `Linear(55, 224)` and ends with:

```text
Linear(416, 320) -> ReLU
Linear(320, 192) -> ReLU
Linear(192, 48)  -> ReLU
Linear(48, 1)    -> ReLU
```

This is better understood as a very large hand-compiled piecewise-linear circuit than as a
conventional language model.

## Final Predicate Circuit

Let `h` be the 192-dimensional nonnegative output entering `Linear(192, 48)`. Its coordinates are
used in 24 blocks of eight. Define the bit-weighted block values

```text
q[k] = sum(2^b * h[8*k + b] for b in 0..7),  k in 0..23
```

The penultimate linear layer forms 16 scalar expressions:

| Predicate | Expression | Target |
|---:|---|---:|
| 0 | `q[0]` | 199 |
| 1 | `q[1]` | 239 |
| 2 | `q[2]` | 101 |
| 3 | `q[3]` | 35 |
| 4 | `q[4] - 2*q[8]` | 60 |
| 5 | `q[5] - 2*q[9]` | 64 |
| 6 | `q[6] - 2*q[10]` | 170 |
| 7 | `q[7] - 2*q[11]` | 50 |
| 8 | `q[12]` | 194 |
| 9 | `q[13]` | 185 |
| 10 | `q[14]` | 172 |
| 11 | `q[15]` | 227 |
| 12 | `q[16] - 2*q[20]` | 117 |
| 13 | `q[17] - 2*q[21]` | 149 |
| 14 | `q[18] - 2*q[22]` | 250 |
| 15 | `q[19] - 2*q[23]` | 124 |

For each expression `v[j]` with target `t[j]`, the 48-unit layer creates three ReLU values:

```text
ReLU(v[j] - (t[j] + 1))
ReLU(v[j] - t[j])
ReLU(v[j] - (t[j] - 1))
```

The last linear layer has exactly 48 weights:

```text
[1] * 16 + [-2] * 16 + [1] * 16
```

and bias `-15`. Therefore it computes:

```text
score = ReLU(
    sum(
        ReLU(v[j] - (t[j] + 1))
        - 2*ReLU(v[j] - t[j])
        + ReLU(v[j] - (t[j] - 1))
        for j in 0..15
    )
    - 15
)
```

For integer-valued `v[j]`, each three-ReLU expression is exactly one when `v[j] == t[j]` and zero
otherwise. The last two parameterized layers therefore implement 16 equality indicators followed
by an AND-like threshold: the final output is one only when all 16 predicates hold.

The weight structure strongly indicates that the earlier 2,719 linear stages compile the 55 input
code points into discrete bit-like intermediate values. Behavioral and activation probes are still
required to map those predicates back to human-readable properties of the text.

## Reproducing the Report

Create an exact Python 3.11 virtual environment, install `requirements/analysis-cpu.txt`, and run:

```bash
python3 scripts/run_architecture_sandbox.py --venv /path/to/python-3.11-venv
```

The validated machine-readable report is written to
`outputs/reports/architecture_report.json`. Generated JSON reports and model artifacts remain
excluded from Git.
