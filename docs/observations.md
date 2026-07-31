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
ordinary MD5 interpretation for the tested ASCII path. Non-ASCII inputs and boundary cases still
need activation capture before generalizing the byte-encoding claim.
