---
name: analyze-relu-local-geometry
description: Analyze Jacobians, Hessian sanity checks, ReLU activation-region boundaries, and steering-direction geometry for this repository's recovered h192-to-output circuit. Use when implementing or reviewing local-geometry reports, measuring semantic-direction effects, locating predicate or final-output ReLU crossings, comparing analytic and autograd derivatives, or deciding whether curvature packages are appropriate.
---

# Analyze ReLU Local Geometry

Treat the recovered tail as a piecewise-affine circuit. Prioritize exact predicate Jacobians,
boundary distances, directional crossing locations, and Jacobian jumps. Use an ordinary output
Hessian only as a sanity check: it is zero within each ReLU region and PyTorch's derivative
convention does not expose a distributional impulse at a kink.

## Read the Project Evidence First

Read these files before changing an analysis or making a claim:

- `docs/semantic_directions_and_local_geometry.md`
- `docs/representation_and_causal_analysis.md`
- `scripts/activation_schema.py`
- `scripts/intervention_schema.py`

Preserve the artifact safety boundary. Never call `torch.load` on `model_3_11.pt` from an ordinary
process. Use the namespace and Landlock launchers for any live model execution. Perform derivative
and geometry analysis offline from validated reports or from a pure-Torch reconstruction of the
already recovered final circuit.

## Use the Exact Recovered Tail

Use the established module roles and do not rename `a48` and `z48` based on conventional notation:

```text
5437 ReLU             -> h192
5438 Linear(192, 48)  -> a48 (preactivation)
5439 ReLU             -> z48
5440 Linear(48, 1)    -> readout preactivation
5441 ReLU             -> scalar output
```

For bit weights `w = [1, 2, 4, 8, 16, 32, 64, 128]`, define 24 block values:

```text
q[k] = dot(w, h192[8*k:8*k+8])
```

Define the 16 predicate values exactly as implemented by
`scripts.activation_schema.predicate_values_from_q`:

```text
p[0:4]   = q[0:4]
p[4:8]   = q[4:8]   - 2*q[8:12]
p[8:12]  = q[12:16]
p[12:16] = q[16:20] - 2*q[20:24]
```

Let `t` be `PREDICATE_TARGETS`. The remaining tail is:

```text
a48 = concat(p - (t + 1), p - t, p - (t - 1))
z48 = relu(a48)
indicator[i] = z48[i] - 2*z48[16+i] + z48[32+i]
readout = sum(indicator) - 15
output = relu(readout)
```

Implement transformed functions using only PyTorch tensor operations. Avoid hooks, mutation, and
NumPy inside functions passed to `torch.func`.

## Compute the Predicate Jacobian Analytically

Build the exact constant matrix `Jp` with shape `16 x 192` from the block equations above. Each
positive block contributes `w`; each subtracted block contributes `-2*w`; all other entries are
zero. This is preferable to differentiating the untrusted full artifact.

Verify the construction independently:

```python
from torch.func import jacrev

Jp_autograd = jacrev(predicate_fn)(h192)
torch.testing.assert_close(Jp_autograd, Jp_analytic)
```

For the 30 candidate semantic directions stacked as `D` with shape `30 x 192`, emit:

```text
direction Gram:       D @ D.T
predicate responses: D @ Jp.T
response Gram:       (D @ Jp.T) @ (D @ Jp.T).T
```

Keep the current semantic result labeled as a null result. Geometry can characterize candidate
directions without establishing that they encode semantics.

## Model Cross-Fold Direction Stability

Use all cross-fold category pairings rather than reporting only the aligned diagonal. For each of
the two slots and three fold pairs, construct a `5 x 5` cosine matrix. This gives six blocks and
150 observations: 30 same-category cells and 120 different-category controls.

Fit the descriptive blocked model:

```text
cosine ~ slot-by-fold-pair fixed effects + same-category indicator
```

Because every block contains five diagonal and twenty off-diagonal cells, the same-category
coefficient equals the overall same-category mean minus the overall different-category mean.
Report both means, the contrast, block matrices, per-slot/category ranges, and model R-squared.

Do not use an ordinary regression p-value that assumes the 150 cells are independent. The three
fold-pair matrices share direction vectors. For permutation inference, anchor fold 0 and permute
the category mappings of folds 1 and 2 independently within each slot; reuse those mappings in
the `0-1`, `0-2`, and `1-2` matrices. Report the seed, repetition count, null quantiles, and a
two-sided plus-one-corrected p-value.

Interpret a positive contrast as reproducible category alignment among fitted directions. It is a
stability diagnostic, not held-out classification accuracy, causal evidence, or proof that the
directions encode semantics. Always retain the leakage-free held-out result as the primary
semantic test.

## Measure ReLU Boundaries Exactly

The 48 predicate-ReLU hyperplanes occur at:

```text
p[i] = t[i] + 1
p[i] = t[i]
p[i] = t[i] - 1
```

For predicate row `j = Jp[i]` and threshold `c`, compute the Euclidean distance from `h` as:

```text
abs(p[i](h) - c) / norm(j)
```

For a normalized direction `d`, compute the signed location along `h + alpha*d` as:

```text
alpha = -(p[i](h) - c) / dot(j, d)
```

Treat a zero denominator as parallel. Report all tied or simultaneous crossings. Treat an exact
zero preactivation as `on_boundary`; do not silently classify it as a stable inactive region.

Within a fixed predicate-ReLU region, derive the readout gradient analytically. For each predicate,
the coefficient multiplying `Jp[i]` is the derivative of:

```text
relu(p-(t+1)) - 2*relu(p-t) + relu(p-(t-1))
```

Sum those rows to obtain `grad_readout`. The local final-ReLU plane has candidate distance:

```text
abs(readout) / norm(grad_readout)
```

Label that value `local` and verify that the perpendicular path does not encounter a predicate
boundary first. If `grad_readout` is zero, report no finite local plane distance. The output
gradient is `grad_readout` when `readout > 0` and zero when `readout < 0`; it is not uniquely
defined at `readout == 0`.

## Measure Jacobian Jumps at Crossings

Prefer exact directional crossing locations over an arbitrary fixed epsilon:

1. Normalize the intervention direction.
2. Solve for every finite `alpha` at a predicate boundary.
3. Select the relevant crossing and choose a small `delta` that does not span another boundary.
4. Evaluate `jacrev` at `h + (alpha-delta)*d` and `h + (alpha+delta)*d`.
5. Report the norm and full vector of the Jacobian difference.
6. Record changed `a48 > 0` rows and whether the final readout sign changed.

Use endpoint sign comparisons for existing intervention reports and preserve the repository's
activation convention `(preactivation > 0)`. Report exact-boundary endpoints separately. A
nonzero jump without a corresponding activation-pattern change, or a reported crossing without
the predicted analytic jump, is a validation failure.

## Use Hessians Only for the Right Question

Use `torch.func.hessian` for a `192 x 192` exact sanity check when the chosen function returns one
scalar:

```python
from torch.func import hessian, jacrev

gradient = jacrev(scalar_fn)(h192)
H_output = hessian(scalar_fn)(h192)
```

Inside a fixed activation region, require the readout and output Hessians to be numerically zero.
Do not interpret a zero result at an exact ReLU boundary as proof of smoothness: automatic
differentiation selects an implementation-specific derivative convention at the kink and commonly
returns a zero second derivative.

Do not introduce a loss merely to obtain nonzero curvature. If the experiment defines a scalar
loss `L(h) = ell(f(h))`, distinguish loss curvature from output curvature. Within an affine region:

```text
H_L = J_f.T @ H_ell @ J_f
```

For scalar `f`, this becomes `ell''(f) * grad(f) @ grad(f).T`. Outside an affine region the full
chain rule also contains the output-Hessian term. This model exposes one scalar, not a multiclass
logit vector, so do not apply cross-entropy without a separately justified objective.

Keep parameter-space Hessians, generalized Gauss-Newton matrices, and Fisher matrices explicitly
separate from derivatives with respect to `h192`.

## Choose Dependencies Conservatively

Use the pinned CPU PyTorch environment for pure-Torch Jacobian and Hessian checks. The supported
interface is `torch.func` (`jacrev`, `jacfwd`, and `hessian`); do not add a standalone `functorch`
dependency.

Do not add SciPy, Matplotlib, Seaborn, BackPACK, or PyHessian to the model-execution environment by
default:

- Use SciPy only for optional sparse eigensolvers or statistical routines.
- Use Matplotlib or Seaborn only for a visualization layer after deterministic JSON exists.
- Use BackPACK only for scalable loss or parameter curvature.
- Use PyHessian only for parameter-loss eigenvalues, trace, or spectral-density work.

If optional packages become necessary, add a separate optional requirements file and native-Linux
virtual environment. Do not silently modify `requirements/analysis-cpu.txt`; NumPy is intentionally
absent from the current sandbox environment.

Primary API references:

- <https://docs.pytorch.org/docs/stable/func.api>
- <https://docs.pytorch.org/docs/stable/func.migrating.html>

## Produce a Reproducible Report

Emit a versioned, deterministic, machine-readable report before plots. Bind it to the hashes of
the semantic analysis, activation report, direction vectors, and any configuration. Include:

- dtype and numerical tolerances;
- `Jp`, its construction version, and analytic-versus-autograd maximum error;
- direction and response Gram matrices;
- baseline activation patterns;
- Euclidean and directional boundary distances;
- crossing identities and locations;
- one-sided Jacobians and jump norms;
- output-Hessian norm and maximum absolute entry;
- any separately defined loss-Hessian results; and
- explicit `on_boundary`, parallel, tied-crossing, and zero-gradient states.

Test small matrices with known Gram results, the exact `Jp` sparsity and coefficients, boundary
distance formulas, simultaneous crossings, and tamper rejection. Reconcile aggregate crossing
counts with validated intervention reports before documenting conclusions.
