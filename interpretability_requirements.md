# Interpretability Requirements

## Likely Interpretability Questions

- How is text tokenized or otherwise encoded?
- Does the model use learned word embeddings, character features, or a lookup table?
- How are multiple words combined?
- Is the scalar output a linear projection of a hidden representation?
- What features or semantic directions do the last layers measure?
- Does changing word order, case, punctuation, repetition, or semantic category predictably change the output?

## Implementation Status

Implemented:

- Dependency/environment specification
- Unit tests and generated fixtures
- Model checksum and provenance reporting
- Full architecture extraction
- Last-layer weight analysis
- Input preprocessing and callable-bytecode recovery
- Hardened, no-inference model loading
- Automated behavioral probing
- Versioned experiment datasets
- Scoped final-layer activation capture
- Exact affine/ReLU readout verification
- Intermediate predicate decoding and MD5 comparison
- Whole-lexeme-held-out semantic direction analysis
- Hash-bound additive and replacement interventions at `h192`
- Exact direction, predicate-Jacobian, Gram, spectrum, and ReLU-boundary analysis
- Deterministic findings synthesis, artifact inventory, and generated final report

Not implemented:

- Exact code-point encoding above 255
- Structured target-preimage search
- Regularized-probe and grouped-bootstrap comparisons
