# Jane Street Steering Vectors

This repository is for investigating the Jane Street puzzle model published on Hugging Face:

- Puzzle Space: https://huggingface.co/spaces/jane-street/puzzle
- Model artifact repo: https://huggingface.co/jane-street/2025-03-10/tree/main

The puzzle presents a PyTorch model artifact and a simple text input interface. Python 3.11 uses `model_3_11.pt`; older Python versions use `model.pt`. The Space sends a raw string directly into the model and returns a numeric output. The hint on the puzzle page suggests starting by looking at the last two layers.

The goal of this project is to reverse engineer what the model computes from text input. In practical terms, that means safely inspecting the serialized PyTorch archive, understanding how the text is represented, identifying the important model layers, and designing experiments that reveal the scoring rule or hidden transformation behind the returned number.

## Current Findings

The model is a 2,721-stage `Linear -> ReLU` circuit with 288,998,553 parameters. It converts the
first 55 characters into Unicode code points, null-pads shorter text, and processes the resulting
float32 vector without a tokenizer or learned input embedding. The final circuit checks 16 exact
integer predicates and returns a positive result only when all 16 hold. See
`docs/architecture_findings.md` for the recovered equations and evidence.

The model file remains untrusted serialized input. Direct pickle loading is disabled; live
architecture analysis is available only through the hash-gated namespace and Landlock launcher.

The current inspection script can:

- Read the `.pt` file as a ZIP archive.
- Compute its SHA-256 digest using bounded memory.
- Record artifact provenance and compare the digest with a trusted value.
- Locate `puzzle/data.pkl` and raw tensor storage files under `puzzle/data/`.
- Parse pickle opcodes with `pickletools` without executing the pickle payload.
- Summarize persistent storage references and map them back to archive byte sizes.
- Report useful metadata such as storage counts, storage types, storage devices, and sample tensor-like references.
- Export the full module tree, all parameter metadata, callable bytecode, the final two leaf
  modules, and the final two parameterized modules inside a restricted worker.
- Enforce exact artifact/runtime compatibility and reject Python 3.12 for Python 3.11 bytecode.
- Deny network access and non-allow-listed filesystem access during live loading.

## Repository Layout

```text
.
├── planning.md
├── MILESTONES.md
├── ARTIFACTS.md
├── pyproject.toml
├── requirements/
│   └── analysis-cpu.txt
├── docs/
│   └── architecture_findings.md
├── scripts/
│   ├── architecture_worker.py
│   ├── inspect_model.py
│   ├── landlock_exec.py
│   ├── run_architecture_sandbox.py
│   └── sandbox_entry.sh
├── src/
│   └── jsmi/
│       └── __init__.py
├── outputs/
│   └── reports/
└── tests/
    ├── fixtures.py
    ├── test_architecture_sandbox.py
    ├── test_inspect_model.py
    └── test_landlock_exec.py
```

All `model*.pt` artifacts are intentionally ignored by Git because they are large local files.

## Environment Setup

The static inspection path supports Python 3.10 or newer and has no third-party runtime
dependencies. The cloudpickled input wrapper in `model_3_11.pt` requires an exact Python 3.11
runtime for live analysis. Create the environment with a Python 3.11 interpreter:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements/analysis-cpu.txt
```

Installing PyTorch does not make loading an untrusted pickle safe. Never invoke `torch.load`
directly on this artifact.

## Running the Safe Inspector

From the repository root:

```bash
python3 scripts/inspect_model.py
```

The default file is selected from the running interpreter: `model_3_11.pt` for Python 3.11 or
newer, otherwise `model.pt`. Use `--model path/to/file.pt` to override it.

To write a JSON report:

```bash
python3 scripts/inspect_model.py \
  --model path/to/model.pt \
  --report outputs/reports/archive_metadata.json
```

By default, this does not execute the pickle payload.

To enforce a supplied digest during static inspection:

```bash
python3 scripts/inspect_model.py \
  --model path/to/model.pt \
  --expected-sha256 <64-hex-character-digest> \
  --report outputs/reports/archive_metadata.json
```

A checksum mismatch returns exit status `2`. The official `model_3_11.pt` digest is built into the
inspector and verified automatically. The old `--load-pickle` path is disabled.

## Running Hardened Architecture Recovery

On Linux with user namespaces and Landlock ABI 3 or newer:

```bash
python3 scripts/run_architecture_sandbox.py
```

The launcher verifies the artifact hash, checks for exact Python 3.11 compatibility, copies the
model into a private staging directory, and starts the worker with isolated namespaces, a Landlock
filesystem allow-list, dropped capabilities, `no_new_privs`, an empty environment, and resource
limits. The worker re-verifies the staged model before unpickling and does not run inference.

## Tests

The tests generate a tiny, non-executable PyTorch-shaped ZIP archive and do not require the real
1.16 GB model or PyTorch:

```bash
python3 -m unittest discover -s tests -v
```

## Milestone Status

Completed:

- Specify the project and pin the optional CPU-only PyTorch environment.
- Add streamed artifact hashing, provenance reporting, and trusted-checksum enforcement.
- Add tests for persistent IDs, `STACK_GLOBAL`, archive metadata, and checksum behavior.
- Implement and validate the hardened architecture-recovery boundary.
- Recover the full architecture, input encoding lambda, and last-layer predicate circuit.

Next:

- Add a probing script that sends simple text inputs to the model and records outputs.
- Design experiments for word order, capitalization, punctuation, repeated words, single words, and word pairs.
- Compare outputs across semantic categories such as animals, foods, colors, places, and verbs.
- Determine whether the model is using embeddings, handcrafted token features, a lookup table, or a learned scoring function.
- Document hypotheses and rejected explanations in a research log.
- Add a reproducible notebook or script for summarizing findings once the model behavior is understood.

See `MILESTONES.md` for the staged implementation plan and acceptance criteria.

## Safety Notes

Do not run `torch.load` directly. PyTorch `.pt` files can execute code during unpickling. Use the
archive-only inspector for static work and `run_architecture_sandbox.py` for the narrowly scoped
architecture export. The sandbox requires Linux and fails closed when namespaces, Landlock,
runtime compatibility, hash verification, or resource controls cannot be established.
