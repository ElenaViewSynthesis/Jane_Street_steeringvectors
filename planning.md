# Jane Street Puzzle Plan

## Source material

- Puzzle Space: https://huggingface.co/spaces/jane-street/puzzle
- Model repo: https://huggingface.co/jane-street/2025-03-10/tree/main

## Problem specification

The puzzle Space loads a PyTorch model from the Hugging Face repo `jane-street/2025-03-10` and exposes a text input box. The app sends the raw input string directly into the model and returns a numeric output.

The prompt text in the Space says:

- "Today I went on a hike and found a pile of tensors..."
- "I'm not sure what it does yet..."
- "Maybe start by looking at the last two layers."

From the current `app.py` in the Space:

- The repo ID is `jane-street/2025-03-10`
- The loaded file is `model.pt`
- Inference is `model(text)`
- The example input is `vegetable dog`

This means the practical objective for this project is to reverse engineer or characterize what `model.pt` computes from text input, likely by inspecting architecture, weights, and behavior on carefully chosen prompts.

## Immediate project goals

1. Download `model.pt` locally without executing it implicitly in unknown code paths.
2. Inspect the file safely and record metadata such as size, hash, and loading requirements.
3. Build a controlled analysis script that can load the model in an isolated environment.
4. Investigate the model architecture, especially the last two layers as hinted by the puzzle.
5. Probe the input/output behavior to infer the text transformation or scoring rule.

## Suggested workflow

1. Verify the downloaded artifact and compute checksums.
2. Create a small inspection script to load the model on CPU.
3. Print the model class, module tree, and final layers.
4. Test simple two-word inputs and look for patterns in outputs.
5. Document hypotheses and eliminate them systematically.

## Safety notes

- The Hugging Face repo marks `model.pt` as a pickle-based artifact.
- Treat the file as untrusted serialized code/data.
- Do not load it outside a controlled Python environment.
- Keep any exploratory loader scripts minimal and auditable.

## Deliverables for this setup step

- `planning.md`
- local copy of `model.pt`
- basic provenance notes for where the file came from
