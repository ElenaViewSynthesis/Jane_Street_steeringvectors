# Jane Street Steering Vectors

This repository is for investigating the Jane Street puzzle model published on Hugging Face:

- Puzzle Space: https://huggingface.co/spaces/jane-street/puzzle
- Model artifact repo: https://huggingface.co/jane-street/2025-03-10/tree/main

The puzzle presents a PyTorch model artifact, `model.pt`, and a simple text input interface. The Space sends a raw string directly into the model and returns a numeric output. The hint on the puzzle page suggests starting by looking at the last two layers.

The goal of this project is to reverse engineer what the model computes from text input. In practical terms, that means safely inspecting the serialized PyTorch archive, understanding how the text is represented, identifying the important model layers, and designing experiments that reveal the scoring rule or hidden transformation behind the returned number.

## Current Focus

The model file is pickle-based, so it should be treated as untrusted serialized input. The first stage of this project focuses on safe static inspection before any live model loading.

The current inspection script can:

- Read the `.pt` file as a ZIP archive.
- Locate `puzzle/data.pkl` and raw tensor storage files under `puzzle/data/`.
- Parse pickle opcodes with `pickletools` without executing the pickle payload.
- Summarize persistent storage references and map them back to archive byte sizes.
- Report useful metadata such as storage counts, storage types, storage devices, and sample tensor-like references.
- Keep live `torch.load` behavior behind an explicit opt-in flag.

## Repository Layout

```text
.
├── planning.md
├── scripts/
│   └── inspect_model.py
├── src/
│   └── jsmi/
│       └── __init__.py
└── outputs/
    └── reports/
```

`model.pt` is intentionally ignored by Git because it is a large local artifact.

## Running the Safe Inspector

From the repository root:

```bash
python scripts/inspect_model.py --model path/to/model.pt
```

To write a JSON report:

```bash
python scripts/inspect_model.py --model path/to/model.pt --report outputs/reports/archive_metadata.json
```

By default, this does not execute the pickle payload.

## TODO

- Install and pin the Python analysis dependencies, including a CPU-only PyTorch setup.
- Add a `requirements.txt` or `pyproject.toml` for reproducible local setup.
- Add unit tests for the pickle opcode parser, especially persistent IDs and `STACK_GLOBAL` handling.
- Add a small fixture-based test archive so tests do not require the 1.16 GB `model.pt` file.
- Build a controlled CPU-only loader that records the model class, module tree, parameter shapes, and final layers.
- Inspect the last two layers in detail, following the puzzle hint.
- Add a probing script that sends simple text inputs to the model and records outputs.
- Design experiments for word order, capitalization, punctuation, repeated words, single words, and word pairs.
- Compare outputs across semantic categories such as animals, foods, colors, places, and verbs.
- Determine whether the model is using embeddings, handcrafted token features, a lookup table, or a learned scoring function.
- Document hypotheses and rejected explanations in a research log.
- Add a reproducible notebook or script for summarizing findings once the model behavior is understood.

## Safety Notes

Do not run `torch.load(model.pt)` casually. PyTorch `.pt` files can contain pickle payloads, and pickle can execute code during loading. Use the archive-only inspection path first, and only use live loading in a controlled environment when you explicitly intend to do so.
