# Mechanistic Observations

## Final Representation and Readout

The readout preactivation is affine in `z48`; the published scalar is not globally linear because
of the final ReLU.

- `5437`: ReLU producing the 192-dimensional representation `h`
- `5438`: `Linear(192, 48)` predicate preactivations
- `5439`: ReLU producing the 48 predicate activations
- `5440`: `Linear(48, 1)` readout preactivation
- `5441`: final ReLU/scalar output

The equations are already recovered in `docs/architecture_findings.md`. Milestone 4 should capture
all four stages so it can verify:

```text
a48 = W5438 h192 + b5438
z48 = ReLU(a48)
readout = W5440 z48 + b5440
output = ReLU(readout)
```

## Activation Smoke Result

The `m4-capture-smoke-v1` suite captured these stages for the empty string, `abc`,
`vegetable dog`, and `red fox`, with two complete repetitions per input. All captures were
bit-deterministic. Recomputing each downstream affine or ReLU operation from the captured tensors
gave a maximum absolute error of `0.0` in every observation.

For all four ASCII inputs, the 16 decoded predicate values exactly equal the ordinary MD5 digest
of the unpadded input:

| Input | Decoded predicate bytes / ordinary MD5 |
|---|---|
| empty string | `d41d8cd98f00b204e9800998ecf8427e` |
| `abc` | `900150983cd24fb0d6963f7d28e17f72` |
| `vegetable dog` | `ab981aaa62cf6412f3aef1a11cd9b94b` |
| `red fox` | `886ad9f73388afe14f2fe4ba1884a2d6` |

None equals the MD5 digest of the corresponding 55-byte null-padded input. This establishes the
ordinary MD5 interpretation for the tested ASCII path.

## MD5 Boundary Result

`m4-md5-boundary-v1` adds 54/55/56-character inputs, embedded and trailing nulls, Latin-1,
combining Unicode, BMP Unicode, and emoji. All 20 observations were deterministic.

- Length 55 and 56 decode to the same predicate digest, confirming a 55-Python-character cutoff.
- A trailing null in `abc\0` decodes to the digest for `abc`; an embedded null has a different
  digest. The model therefore exposes a null-sensitive input path with nontrivial termination
  behavior.
- `café` and `ÿ` agree with one-byte Latin-1/code-point candidates, while decomposed `café`,
  `漢`, and `😀` do not agree with any tested UTF-8, Latin-1, truncated, or padded candidate.

Thus the recovered circuit is MD5-like on the tested short ASCII and byte-range paths, but the
exact universal conversion from the 55 float code points to MD5 bytes remains unresolved. In
particular, it is not a universal raw UTF-8, raw Latin-1, or 55-byte-null-padded encoding rule.

## Local Geometry Result

The exact predicate Jacobian from `h192` is a constant `16 x 192` matrix with 192 nonzero entries.
It agrees exactly with `torch.func.jacrev`. The 30 fitted semantic directions span numerical rank
24, with entropy effective rank approximately 19.656.

Across the six slot/fold-pair blocks, the full 150-cell cosine model compares 30 same-category
pairs (mean 0.247) with 120 different-category controls (mean -0.062). The blocked
same-category contrast is 0.309 with R-squared 0.524. A coherent fold-label permutation test
(10,000 repetitions, fold 0 anchored) gives two-sided p=0.0001. This is evidence of repeatable
direction-label alignment, not evidence of held-out semantic prediction; held-out accuracy remains
at chance.

Across 600 unique nonzero direction/case endpoints, 265 change at least one predicate-ReLU
activation pattern. Repetition produces the 530 crossing observations already present in the
semantic intervention report; the offline geometry analysis predicts every crossing with zero
mismatches. No endpoint crosses the final output ReLU.

At a clean fixed-region baseline, ordinary readout and output Hessians are both exactly zero.
This is a piecewise-affine result, not evidence that the circuit lacks nonlinear boundaries.
Jacobian jumps and explicit ReLU crossings are the relevant measurements.
