# Hypothesis and Falsification Log

This log records the principal mechanistic hypotheses, the evidence used to test them, and the
remaining scope of each conclusion. Generated aggregate values are checked by
`scripts/synthesize_findings.py`.

| Hypothesis | Test | Result | Status |
|---|---|---|---|
| The scalar exposes smooth semantic variation. | Run controlled and semantic-factorial scalar probes twice per input. | All 514 observations are deterministic zeros. | Falsified for the tested domain. |
| The final representation carries ordinary MD5 bytes for short ASCII. | Decode the 16 predicates from `h192` and compare named digest candidates. | All tested short ASCII cases equal ordinary unpadded-input MD5. | Supported on the tested path. |
| The circuit hashes all Unicode through UTF-8. | Compare Latin-1, decomposed Unicode, BMP, and emoji boundary cases. | Wider Unicode cases do not match the tested UTF-8 candidates. | Falsified as a universal rule. |
| The input has no fixed character cutoff. | Compare 55 and 56 copies of `a`. | Both decode identically. | Falsified; the cutoff is 55 Python characters. |
| Mean-difference semantic directions generalize to unseen lexemes. | Hold out one whole lexeme per category and exclude every pair containing a held-out word. | Left accuracy 0.180 and right 0.207 versus 0.200 chance and randomized controls. | Null result; not supported. |
| The recovered final gate is only correlational. | Replace `h192` with the canonical target and sixteen single-predicate breaks. | Target gives output 1; every break gives output 0. | Falsified; the recovered gate is causal. |
| A first-order predicate map only approximates interventions. | Compare analytic and observed predicate deltas across 1,800 observations. | Maximum predicate-delta error is 0. | Falsified; the predicate map is exactly affine up to represented arithmetic. |
| An ordinary Hessian captures the important local nonlinearity. | Compute `torch.func` Hessians at a clean fixed-region point and enumerate ReLU crossings. | Hessians are zero while 530 observations cross predicate ReLUs. | Falsified; Jacobian jumps and boundary crossings are primary. |
| PyHessian is required for `h192` geometry. | Compare its parameter-loss scope with the recovered activation-space question. | No parameter-space loss objective exists; the exact `192 x 192` activation Hessian is already tractable. | Not applicable to the current milestone. |

## Open hypotheses

- Code points above 255 may be mapped by a thresholded, decomposed, or sentinel-based encoding;
  the existing boundary suite does not distinguish these possibilities.
- The target digest may have a preimage in a finite puzzle-specific phrase grammar, but an
  unrestricted MD5 preimage search is infeasible.
- Alternative regularized probes may characterize lexical geometry, but they must use the same
  whole-lexeme exclusions and cannot overturn the current estimator's null result without new
  leakage-free evidence.
