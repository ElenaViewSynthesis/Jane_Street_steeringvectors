# Interpretability Requirements

## Likely Interpretability Questions

- How is text tokenized or otherwise encoded?
- Does the model use learned word embeddings, character features, or a lookup table?
- How are multiple words combined?
- Is the scalar output a linear projection of a hidden representation?
- What features or semantic directions do the last layers measure?
- Does changing word order, case, punctuation, repetition, or semantic category predictably change the output?

## Not Implemented

- Dependency/environment specification
- Unit tests or fixtures
- Model checksum and provenance reporting
- Full architecture extraction
- Activation capture or hooks
- Last-layer weight analysis
- Automated behavioral probing
- Experiment datasets
- Hypothesis/research logging
- Steering-vector extraction or intervention
- Reproducible notebook/report
