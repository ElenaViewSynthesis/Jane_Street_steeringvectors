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

Not implemented:

- Activation capture or hooks
- Hypothesis/research logging
- Steering-vector extraction or intervention
- Reproducible notebook/report
