# Behavioral Probing Findings

## Execution Protocol

Milestone 3 executes model inference only inside a dedicated probe worker. The architecture worker
remains a no-inference path. The probe launcher applies the same core isolation controls used for
architecture recovery:

- trusted model SHA-256 verification before staging and immediately before `torch.load`;
- exact Python 3.11 artifact compatibility;
- isolated user, mount, PID, network, IPC, and UTS namespaces;
- a Landlock filesystem allow-list, dropped capabilities, and `no_new_privs`;
- read-only staged model and manifest files;
- a sanitized environment and CPU, memory, file, descriptor, and process limits; and
- bounded, symlink-resistant, atomically published JSON output.

Probe manifests have a versioned schema, unique case identifiers, bounded string inputs, and flat
experimental factors. The host validates a manifest before entering the sandbox. The worker then
revalidates it and checks the exact manifest-file SHA-256 before inference.

The worker loads the model once, sets evaluation and inference modes, enables deterministic PyTorch
algorithms, fixes the seed at zero, and uses one CPU thread. Each suite is run as two complete
passes, so repeated observations are separated by all other cases rather than called adjacently.
The host accepts a report only when every result is bound to the expected model hash, manifest
hash, case identifier, exact input, repetition, tensor shape, dtype, and finite scalar value.

## Experiment Suites

### Smoke and controlled contrasts

`experiments/probes/m3-smoke-v1.json` contains 32 inputs covering:

- empty, single-character, whitespace, and the documented `vegetable dog` example;
- implicit versus explicit null padding;
- lengths 54, 55, and 56 and suffixes beyond the 55-character boundary; and
- controlled word order, uppercase, title case, punctuation, repetition, and whitespace changes.

Manifest SHA-256:

```text
fbabfb40471ba1ae967aa77269029c8df0c7a48c9e244328e2a91532592b25f8
```

The two complete passes produced 64 bit-identical observations. Every output was `0.0` with Python
float encoding `0x0.0p+0`.

### Semantic factorial

`experiments/probes/m3-semantic-factorial-v1.json` crosses 15 words in both positions, using three
members from each category:

- animals: `cat`, `dog`, `fox`;
- foods: `apple`, `bread`, `carrot`;
- colors: `blue`, `green`, `red`;
- places: `london`, `paris`, `rome`; and
- verbs: `jump`, `run`, `sleep`.

The complete ordered design contains 225 pairs, including same-word and same-category cases.

Manifest SHA-256:

```text
4d63fb53c01f7ce8cf21e33fe052cc4898a87e067141e35d275cf850ce939e33
```

The two complete passes produced 450 bit-identical observations. Every output was again `0.0`.

## Interpretation

The final scalar contains no variance over the 257 tested inputs. Consequently:

- no order, capitalization, punctuation, repetition, whitespace, or tested length effect is
  visible at the final output;
- no animal, food, color, place, or verb main effect is estimable; and
- no lexical, positional, or category interaction is estimable from these scalar results.

This is a null result, not evidence that the model ignores those properties. Architecture recovery
showed that the output is an AND-like gate over 16 exact equality predicates. A probe that misses
even one predicate returns zero, so radically different internal states can collapse to the same
scalar output.

The architecture remains consistent with a one-block hash-style circuit: the model accepts 55
character positions and checks 16 byte-sized targets. MD5 is therefore a high-priority hypothesis,
but scalar probing has not proven it. The ordinary MD5 of `vegetable dog` and the MD5 of its
55-byte null-padded ASCII form are respectively:

```text
ab981aaa62cf6412f3aef1a11cd9b94b
49cdcb2281663f1ffed82adee6c57160
```

Neither equals the recovered target `c7ef65233c40aa32c2b9ace37595fa7c`.

## Reproduction

Regenerate the committed manifests:

```bash
python3 scripts/generate_probe_manifests.py
```

Run the smoke suite with an exact Python 3.11 analysis environment:

```bash
python3 scripts/run_probe_sandbox.py \
  --venv /path/to/python-3.11-venv \
  --manifest experiments/probes/m3-smoke-v1.json \
  --report outputs/probes/m3-smoke-v1.json \
  --repetitions 2
```

Run the semantic factorial:

```bash
python3 scripts/run_probe_sandbox.py \
  --venv /path/to/python-3.11-venv \
  --manifest experiments/probes/m3-semantic-factorial-v1.json \
  --report outputs/probes/m3-semantic-factorial-v1.json \
  --repetitions 2
```

Generated probe reports are excluded from Git because they contain platform-specific runtime
metadata. Their manifest hashes and aggregate results are recorded above.

## Next Research Boundary

Additional broad scalar probes are unlikely to be informative until a positive input is known.
Milestone 4 should capture the 192-dimensional representation entering the predicate layer and the
48 predicate activations. Comparing the decoded intermediate bytes with candidate MD5 digests can
directly test the hash hypothesis and reveal how many predicates each input satisfies.

That follow-up has now begun. The first Milestone 4 activation suite found that the 16 decoded
predicate values exactly equal ordinary MD5 for all four tested ASCII inputs. See
`docs/representation_and_causal_analysis.md` for the scoped-hook protocol and current evidence.
