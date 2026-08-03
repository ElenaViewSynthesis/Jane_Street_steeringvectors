# Glossary

## Canonical representation

A canonical representation is one standardized, unambiguous encoding of a value. If two values
are considered identical for an experiment, their canonical representations should also be
identical. Canonical encodings make hashes, comparisons, and reproducibility checks meaningful by
removing irrelevant choices such as formatting or platform byte order.

This project uses the term in two specific ways:

- **Canonical tensor representation:** activation values are converted to finite CPU float32
  values and encoded as little-endian float32 bytes before SHA-256 hashing. This gives one stable
  byte representation for comparing repeated captures.
- **Canonical target representation:** a deliberately constructed 192-dimensional `h192` vector
  whose 24 eight-bit blocks satisfy all 16 recovered predicate targets. Replacing a naturally
  produced `h192` with this standardized target state is a causal intervention. It produces a
  positive final output; changing one target predicate at a time turns that output off.

"Canonical" does not mean that the model naturally uses the only possible representation. Many
different 192-dimensional states could satisfy the same 16 equations. It means the experiment
chooses one documented representative so repeated interventions are directly comparable.

## Crossing observation

A crossing observation is one repeated intervention result in which at least one of the 48
predicate preactivations changes its active/inactive state between the baseline and the
intervention endpoint. It is counted once even if several predicate-ReLU rows cross during that
observation.

The semantic sweep contains 30 directions, 10 cases per direction, three strengths, and two
repetitions. The two nonzero strengths therefore define 600 unique direction/case endpoints and
1,200 repeated endpoint observations. Of the unique endpoints, 265 cross at least one predicate
ReLU. Two repetitions of each give the reported 530 crossing observations. Strength-zero results
do not cross a boundary.

"Reproduced with zero mismatches" means the offline geometry analyzer used the exact recovered
predicate equations, float32 intervention vectors, baseline activations, and `> 0` activation
convention to predict the complete set of changed ReLU rows for every endpoint. It then compared
those predictions with the already validated intervention report. Every predicted row set matched
both recorded repetitions, so the analytic count of 530 equaled the observed count of 530.

This agreement validates the recovered final-circuit geometry and its numerical convention over
the tested endpoints. It does not show that the directions are semantic, does not constitute a
new model-inference run, and does not make a claim about untested boundaries earlier in the
network.

## Cross-fold cosine similarity

Cross-fold cosine similarity measures whether independently fitted directions for the same slot
and semantic category point in a consistent direction when different lexemes are held out. For
L2-normalized directions `d_i` and `d_j`, it is their dot product:

```text
cosine(d_i, d_j) = d_i dot d_j
```

Values near `1` indicate alignment, values near `-1` indicate opposition, and values near `0`
indicate little linear alignment. This project compares each pair of the three whole-lexeme folds
for each of five categories and two input slots, producing 30 same-slot/category cross-fold
comparisons. Their mean is approximately `0.247`, which indicates modest positive alignment rather
than a stable shared direction.

The implemented model uses all 150 cells from six `5 x 5` slot/fold-pair cosine matrices, not just
the aligned diagonal. It compares 30 same-category cells (mean approximately `0.247`) with 120
different-category controls (mean approximately `-0.062`) while including a fixed effect for each
slot/fold-pair block. The same-category contrast is approximately `0.309`. Its permutation test
shuffles category mappings coherently across the three folds within each slot, preserving the
dependence created by shared direction vectors. None of 10,000 seeded permutations was as extreme
as the observed contrast, giving a plus-one-corrected two-sided p-value of approximately `0.0001`.

Cross-fold cosine is a stability diagnostic, not a classification score. A significant alignment
contrast can show that independently fitted direction labels recur more than arbitrary category
pairings, but it cannot by itself establish semantic content or held-out predictive utility. The
separate same-fold/category comparison between left and right slots has mean cosine approximately
`-0.081`, and held-out category accuracy remains at chance.

## cloudpickle

`cloudpickle` is a Python serialization library built on the standard `pickle` protocol. It can
serialize more dynamic Python objects than ordinary `pickle`, including lambdas, locally defined
functions, closures, and their code objects.

The puzzle artifact uses cloudpickled Python code for its custom input wrapper. That is why
`model_3_11.pt` requires a compatible Python 3.11 runtime: serialized Python bytecode is tied to
the interpreter version that created it.

Loading cloudpickle data is unsafe when the source is untrusted. Deserialization can import
modules and execute arbitrary Python code. In this repository, `cloudpickle` is therefore used
only inside the hash-gated namespace and Landlock sandbox; static archive inspection never
unpickles the artifact.

## Entropy effective rank

Entropy effective rank is a smooth estimate of how many dimensions materially contribute to a
matrix. For singular values `s_i`, first convert squared singular values into normalized energy
weights:

```text
p_i = s_i^2 / sum_j(s_j^2)
```

Then compute the exponential of their Shannon entropy:

```text
effective_rank = exp(-sum_i(p_i * log(p_i)))
```

It equals `1` when essentially all energy lies in one singular direction and approaches the
ordinary rank when energy is distributed evenly across all nonzero singular directions. Unlike
numerical rank, it has no hard singular-value cutoff and need not be an integer.

For the `30 x 192` semantic-direction matrix, the numerical rank is 24 while the entropy effective
rank is approximately `19.656`. This means 24 singular directions clear the numerical tolerance,
but their energy is distributed like roughly 19.7 equally weighted dimensions. It describes the
geometry of the candidate direction family; it is not evidence that those dimensions encode
semantics.

## Forward hook

A forward hook is a callback registered on a PyTorch module. PyTorch calls it after that module's
`forward` computation and passes it the module, its inputs, and its output. A hook can observe an
intermediate tensor without rewriting the model's `forward` method.

Milestone 4 installs forward hooks on modules `5437` through `5441` to capture `h192`, `a48`,
`z48`, the readout preactivation, and the final output. The hooks are scoped: they exist only around
the controlled inference loop and their removable handles are always removed when that scope
exits. An intervention hook can return a replacement activation, while the capture hooks only
observe outputs. The separate intervention worker uses a scoped transformation hook to add or
replace `h192` before capturing the effective downstream state.

Hooks are useful but require care. Registering the same hook more than once can duplicate data,
leaving hooks installed can contaminate later experiments, and retaining live tensors can retain
unnecessary memory. The activation worker rejects duplicate firings, snapshots detached CPU
values, and verifies that every hook was removed.

## JSMI

`JSMI` is this repository's local abbreviation for **Jane Street Mechanistic Interpretability**.
It appears in the Python package name (`jsmi`), sandbox profile environment variables such as
`JSMI_SANDBOX_PROFILE`, and temporary-directory prefixes.

Mechanistic interpretability aims to explain a model in terms of its internal computations rather
than only its input/output behavior. Here that means recovering the character encoding, linear and
ReLU circuit, intermediate MD5 bytes, equality predicates, and causal effect of changing internal
activations.

## Predicate-ReLU crossing

Each of the 16 recovered predicate values `p_i` is tested by three ReLU rows with boundaries at
`target_i + 1`, `target_i`, and `target_i - 1`. Together these form the 48-dimensional `a48`
preactivation. A predicate-ReLU crossing occurs when an intervention changes a row's state under
the repository's exact convention:

```text
active = preactivation > 0
```

An endpoint exactly equal to zero is recorded as being on the boundary and is not silently treated
as a stable interior point. A predicate value can move without crossing a ReLU, and crossing one
or more predicate ReLUs does not imply crossing the final output ReLU. In the semantic sweep, 530
repeated observations cross at least one predicate ReLU, while none crosses the final ReLU and all
final scalar outputs remain zero.
