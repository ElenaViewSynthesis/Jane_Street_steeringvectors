# Semantic Directions and Local Geometry

## Purpose

This document connects the Milestone 4 semantic-direction experiment with the local geometry of
the recovered final circuit. It defines `h192`, explains the 30 fitted directions and 1,800
interventions, and distinguishes direction Gram matrices, Jacobians, and Hessians.

## The `h192` Representation

`h192` is the nonnegative 192-dimensional output of module `5437`, immediately before
`Linear(192, 48)` at module `5438`:

```text
5437 ReLU             -> h192
5438 Linear(192, 48)  -> a48
5439 ReLU             -> z48
5440 Linear(48, 1)    -> readout preactivation
5441 ReLU             -> scalar output
```

Its coordinates are divided into 24 blocks of eight. Each block defines a bit-weighted value:

\[
q_k = \sum_{b=0}^{7} 2^b h_{8k+b}, \qquad k=0,\ldots,23
\]

The recovered circuit combines those `q` values into 16 MD5-byte-like predicate expressions,
compares them with fixed digest targets through three-ReLU equality indicators, and publishes a
nonzero scalar only when all 16 predicates match.

## Thirty Semantic Directions

For a slot `s`, held-out fold `f`, and semantic category `c`, the current estimator is the
normalized one-vs-rest mean difference:

\[
d_{s,f,c} =
\frac{\mu_{s,f,c} - \mu_{s,f,\neg c}}
{\left\|\mu_{s,f,c} - \mu_{s,f,\neg c}\right\|_2}
\]

Only training lexemes contribute to the means. Every case containing a held-out lexeme in either
word position is excluded from fitting. The 30 directions are:

\[
2\ \text{input slots} \times 3\ \text{whole-lexeme folds}
\times 5\ \text{categories} = 30
\]

The complete semantic intervention sweep is:

\[
30\ \text{directions} \times 10\ \text{held-out cases each}
\times 3\ \text{strengths}\ (-1,0,+1)
\times 2\ \text{repetitions} = 1800
\]

For an additive intervention with strength `α`, the effective representation is:

\[
h_{after} = h_{baseline} + \alpha d
\]

The observed held-out classification accuracy was approximately chance, so these vectors are
candidate directions rather than validated semantic features. Their causal effects remain useful
for testing the recovered downstream circuit.

## Predicate Deltas

A predicate delta is the 16-dimensional vector:

\[
\Delta p = p(h_{after}) - p(h_{baseline})
\]

It measures movement in the recovered MD5-byte-like predicate expressions. It does not mean that
an equality indicator changed, and it does not imply that the final scalar became nonzero. Those
events depend on crossing predicate-ReLU boundaries and, ultimately, the final ReLU boundary.

The semantic sweep observed predicate deltas from approximately `-40.615` to `+40.615` at
strengths `-1` and `+1`. Predicted and observed predicate deltas agreed exactly. Strength `0`
produced zero deltas. At least one predicate-ReLU boundary was crossed in 530 observations, but no
intervention crossed the final-output ReLU.

## Modeling the Directions as a Matrix

Stack the 30 normalized directions as rows of:

\[
D \in \mathbb{R}^{30\times192}
\]

This representation supports several complementary analyses:

- compare direction stability across folds and input slots;
- detect duplicated or opposing directions;
- estimate the effective rank of the candidate direction family;
- form consensus directions only when fold-specific vectors align; and
- map every direction analytically into the 16 recovered predicate values.

The current code uses standard-library arithmetic to keep the estimator explicit and auditable.
The controlled execution environment is pinned to CPU-only PyTorch and `cloudpickle`; NumPy,
SciPy, and scikit-learn are not currently required.

## Recommended Linear-Probe Comparison

The mean-difference estimator should remain the transparent baseline. A useful regularized
comparison is a one-vs-rest ridge probe:

\[
w^* = \arg\min_w
\sum_i \left(y_i - w^\top h_i\right)^2 + \lambda\lVert w\rVert_2^2
\]

A multiclass ridge classifier produces five coefficient vectors for each slot and fold, retaining
the same `2 × 3 × 5 = 30` design. Each coefficient should be converted back into raw `h192`
coordinates if feature scaling is used, then L2-normalized before intervention. Regularization is
valuable because there are 192 representation coordinates but relatively few independent
lexemes.

Shrinkage linear discriminant analysis is a reasonable secondary comparison because it accounts
for shared within-class covariance. For five classes, however, its discriminant subspace has at
most four dimensions, so it does not naturally preserve five independent one-vs-rest directions
per slot and fold.

Validation must remain lexeme-grouped. A generic row-level split would leak repeated word identity
across training and evaluation. Standard grouped cross-validation is helpful, but paired inputs
contain two lexeme memberships, so the repository's explicit rule—exclude any training case
containing any held-out word—remains the authoritative split.

Randomization and uncertainty estimates must also operate at the lexeme level. Labels should be
permuted across lexeme/category assignments rather than independently across repeated pairs, and
bootstrap resampling should sample lexeme clusters rather than individual pair rows.

## Gram Matrix

A Gram matrix is a table of pairwise dot products. For the normalized direction matrix `D`:

\[
G_D = DD^\top \in \mathbb{R}^{30\times30}
\]

Each entry is:

\[
(G_D)_{ij} = d_i^\top d_j
\]

Because every direction has unit L2 norm, the entries are cosine similarities:

- `1` means two directions are perfectly aligned;
- `-1` means they point in opposite directions;
- `0` means they are orthogonal; and
- intermediate values measure partial alignment.

For this experiment, the Gram matrix reveals whether the same category direction is stable across
folds, whether left- and right-slot directions align, and whether apparently different categories
recover the same lower-dimensional structure. Stable semantic directions should form consistent
blocks in the Gram matrix. A noisy matrix with near-zero cross-fold similarities supports the null
result.

## Jacobian

Let the 16 predicate expressions be a function of `h192`:

\[
p(h): \mathbb{R}^{192} \rightarrow \mathbb{R}^{16}
\]

Their Jacobian is:

\[
J = \frac{\partial p}{\partial h}
\in \mathbb{R}^{16\times192}
\]

Each row describes how one predicate responds to every `h192` coordinate. For a small
intervention `αd`:

\[
\Delta p \approx \alpha Jd
\]

The recovered predicate map is affine, so this relation is exact up to float32 arithmetic rather
than merely a first-order approximation.

For all 30 directions, define their predicate responses as:

\[
R = DJ^\top \in \mathbb{R}^{30\times16}
\]

The response Gram matrix is:

\[
G_R = RR^\top = DJ^\top JD^\top
\]

`G_D` compares directions in representation space. `G_R` compares them after the predicate
circuit. Two directions can be dissimilar in `h192` yet produce similar downstream effects when
their responses `Jd` align.

## Hessian

For a scalar function such as a loss or readout `L(h)`, the Hessian is:

\[
H = \frac{\partial^2 L}{\partial h^2}
\in \mathbb{R}^{192\times192}
\]

The Hessian measures curvature: how the gradient changes as `h192` moves. The recovered model is a
ReLU network and therefore piecewise affine:

- inside one fixed ReLU region, the Jacobian is constant;
- inside that region, the model-output Hessian is zero; and
- at a ReLU boundary, the Jacobian changes discontinuously and an ordinary Hessian is not defined.

This is why intervention reports explicitly record predicate and final-ReLU crossings. The
relevant nonlinearity appears as a change of affine region rather than smooth curvature.

## Jacobian Gram Matrices and Hessian Approximations

The matrix

\[
J^\top J
\]

is a Gram matrix of the Jacobian rows. It is positive semidefinite and measures directions in
`h192` space to which the predicates are most sensitive. It is not generally the true Hessian.

For a least-squares objective, `JᵀJ` is the Gauss–Newton approximation to the Hessian. If the
modeled function is locally affine, the output's second-derivative contribution vanishes, making
the approximation exact for the relevant local loss curvature, apart from conventional scaling
factors.

The principal objects therefore answer different questions:

| Object | Shape | Question |
|---|---:|---|
| `D` | `30 × 192` | What candidate intervention directions were fitted? |
| `DDᵀ` | `30 × 30` | Which candidate directions align in `h192` space? |
| `J` | `16 × 192` | How do predicates respond locally to `h192`? |
| `DJᵀ` | `30 × 16` | How does each candidate direction change the predicates? |
| `DJᵀJDᵀ` | `30 × 30` | Which directions behave similarly downstream? |
| `H` | `192 × 192` | How does a scalar gradient curve within a region? |
| `JᵀJ` | `192 × 192` | What is the Jacobian sensitivity geometry or Gauss–Newton curvature? |

## Recommended Tooling

The existing sandbox and intervention mechanism should continue using the pinned PyTorch runtime.
No transformer-specific steering library is needed because the artifact is a custom
`torch.nn.Sequential` circuit rather than a transformer.

An optional offline direction-analysis environment could add:

- NumPy for direction matrices, Gram matrices, SVD, and vectorized projections;
- scikit-learn for ridge probes, pipelines, grouped cross-validation, metrics, and permutation
  scores; and
- SciPy for permutation procedures and bootstrap confidence intervals.

Relevant primary documentation:

- [PyTorch module forward hooks](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [scikit-learn RidgeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html)
- [scikit-learn LinearDiscriminantAnalysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
- [scikit-learn GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
- [scikit-learn permutation_test_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html)
- [SciPy bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
- [SciPy permutation_test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html)

## Repository References

- `scripts/analyze_activation_report.py`: direction fitting, held-out evaluation, and randomized
  labels.
- `scripts/generate_intervention_specs.py`: conversion of the 30 fitted directions into bounded,
  hash-bound intervention specifications.
- `scripts/intervention_worker.py`: additive and replacement interventions at `h192`.
- `scripts/intervention_schema.py`: analytic predicate/readout predictions and report validation.
- `docs/architecture_findings.md`: recovered `h192` block structure, predicate equations, and
  three-ReLU equality circuit.

## Implemented Local-Geometry Result

`scripts/analyze_local_geometry.py` implements the deterministic analysis in a separate Python
3.11 environment defined by `requirements/geometry-analysis.txt`. PyTorch computes and verifies
the exact derivatives, SciPy independently checks the singular values, and Matplotlib/Seaborn
render the generated heatmaps and spectrum.

The validated report contains:

- the exact `16 x 192` predicate Jacobian with 192 nonzero entries;
- zero maximum error between the analytic matrix and `torch.func.jacrev`;
- the `30 x 30` direction Gram and response Gram matrices;
- numerical direction rank 24 and entropy effective rank approximately 19.656;
- mean same-category cross-fold cosine 0.247 within a slot;
- a six-block, 150-cell cross-fold model in which same-category cosine 0.247 exceeds the
  different-category control mean -0.062 by 0.309;
- a coherent 10,000-repetition fold-label permutation p-value of 0.0001 for that alignment
  contrast;
- mean same-category cross-slot cosine -0.081 within a fold;
- 600 nonzero direction/case endpoints, 265 of which cross a predicate ReLU; and
- exact reproduction of all 530 repeated crossing observations with zero mismatches.

At the clean `pair-apple-bread` baseline, neither predicate preactivation is on a boundary. Both
the readout and output Hessian Frobenius norms are exactly zero, as expected within a fixed ReLU
region. PyHessian is not used: it targets parameter-space loss eigenvalues, trace, and spectral
density, while this analysis differentiates the recovered tail with respect to `h192` and defines
no parameter-space loss.

The cross-fold model shows reproducible alignment among directions fitted under the same category
label. It does not rescue the semantic hypothesis: the held-out classifier remains at chance, and
cross-slot alignment is slightly negative on average. The modeled contrast is therefore reported
as direction-estimation stability, not evidence of semantic representation.

Reproduce the report with the command documented in `README.md`; generated JSON and plots live
under `outputs/reports/` and remain ignored by Git.
