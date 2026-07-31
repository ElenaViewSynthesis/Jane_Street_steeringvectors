# Jane Street Steering Vectors

This repository is for investigating the Jane Street puzzle model published on Hugging Face:

- Puzzle Space: https://huggingface.co/spaces/jane-street/puzzle
- Model artifact repo: https://huggingface.co/jane-street/2025-03-10/tree/main

The puzzle presents a PyTorch model artifact and a simple text input interface. Python 3.11 or newer uses `model_3_11.pt`; older Python versions use `model.pt`. The Space sends a raw string directly into the model and returns a numeric output. The hint on the puzzle page suggests starting by looking at the last two layers.

The goal of this project is to reverse engineer what the model computes from text input. In practical terms, that means safely inspecting the serialized PyTorch archive, understanding how the text is represented, identifying the important model layers, and designing experiments that reveal the scoring rule or hidden transformation behind the returned number.

## Current Focus

The model file is pickle-based, so it should be treated as untrusted serialized input. The first stage of this project focuses on safe static inspection before any live model loading.

The current inspection script can:

- Read the `.pt` file as a ZIP archive.
- Compute its SHA-256 digest using bounded memory.
- Record artifact provenance and compare the digest with a trusted value.
- Locate `puzzle/data.pkl` and raw tensor storage files under `puzzle/data/`.
- Parse pickle opcodes with `pickletools` without executing the pickle payload.
- Summarize persistent storage references and map them back to archive byte sizes.
- Report useful metadata such as storage counts, storage types, storage devices, and sample tensor-like references.
- Export the full module tree, parameter metadata, buffers, and final two leaf modules after an
  explicitly opted-in live load.
- Keep live `torch.load` behavior behind an explicit opt-in flag.

## Repository Layout

```text
.
├── planning.md
├── MILESTONES.md
├── ARTIFACTS.md
├── pyproject.toml
├── requirements/
│   └── analysis-cpu.txt
├── scripts/
│   └── inspect_model.py
├── src/
│   └── jsmi/
│       └── __init__.py
├── outputs/
│   └── reports/
└── tests/
    ├── fixtures.py
    └── test_inspect_model.py
```

All `model*.pt` artifacts are intentionally ignored by Git because they are large local files.

## Environment Setup

The static inspection path supports Python 3.10 or newer and has no third-party runtime
dependencies. Live model loading is optional and uses the pinned CPU-only PyTorch environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements/analysis-cpu.txt
```

Installing PyTorch does not make loading an untrusted pickle safe. Use the live-loading phase only
inside a disposable environment with network access disabled and tightly limited filesystem access.

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

Once an independently trusted digest is available, enforce it before any live load:

```bash
python3 scripts/inspect_model.py \
  --model path/to/model.pt \
  --expected-sha256 <64-hex-character-digest> \
  --report outputs/reports/archive_metadata.json
```

A checksum mismatch returns exit status `2` and prevents the `--load-pickle` phase from starting.
The official `model_3_11.pt` digest is built into the inspector and verified automatically. Other
artifacts require `--expected-sha256` before the live-loading flag will run.

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
- Implement complete module and parameter metadata export for the future controlled-load phase.

Next:

- Run and validate the CPU-only metadata exporter inside a hardened disposable environment.
- Inspect the last two layers in detail, following the puzzle hint.
- Add a probing script that sends simple text inputs to the model and records outputs.
- Design experiments for word order, capitalization, punctuation, repeated words, single words, and word pairs.
- Compare outputs across semantic categories such as animals, foods, colors, places, and verbs.
- Determine whether the model is using embeddings, handcrafted token features, a lookup table, or a learned scoring function.
- Document hypotheses and rejected explanations in a research log.
- Add a reproducible notebook or script for summarizing findings once the model behavior is understood.

See `MILESTONES.md` for the staged implementation plan and acceptance criteria.

## Safety Notes

Do not run `torch.load(model.pt)` casually. PyTorch `.pt` files can contain pickle payloads, and pickle can execute code during loading. The current `--load-pickle` child process limits CPU threading and disables CUDA, but it is not a security sandbox. Use the archive-only inspection path first, and only perform live loading in an externally sandboxed, disposable environment.
