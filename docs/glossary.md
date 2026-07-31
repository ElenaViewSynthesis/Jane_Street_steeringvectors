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
  produced `h192` with this standardized target state is a planned causal intervention. It should
  produce a positive final output; changing one target block at a time should turn that output off.

"Canonical" does not mean that the model naturally uses the only possible representation. Many
different 192-dimensional states could satisfy the same 16 equations. It means the experiment
chooses one documented representative so repeated interventions are directly comparable.

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

## Forward hook

A forward hook is a callback registered on a PyTorch module. PyTorch calls it after that module's
`forward` computation and passes it the module, its inputs, and its output. A hook can observe an
intermediate tensor without rewriting the model's `forward` method.

Milestone 4 installs forward hooks on modules `5437` through `5441` to capture `h192`, `a48`,
`z48`, the readout preactivation, and the final output. The hooks are scoped: they exist only around
the controlled inference loop and their removable handles are always removed when that scope
exits. A future intervention hook may return a replacement activation, but the current capture
hooks only observe outputs.

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
