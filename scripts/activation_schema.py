from __future__ import annotations

import hashlib
import math
import struct
from typing import Any

try:
    from scripts.probe_schema import ProbeSchemaError
except ModuleNotFoundError:  # Direct execution from scripts/.
    from probe_schema import ProbeSchemaError  # type: ignore[no-redef]


REPORT_SCHEMA_VERSION = 1
SANDBOX_PROFILE = "linux-userns-landlock-activation-v1"
CAPTURE_PROFILE = "final-predicate-activations-v1"
# Bit-valued native h192 states reconstruct exactly at float32 precision.  Additive
# steering directions introduce ordinary float32 accumulation error across eight
# weighted bit positions, so keep a conservative numerical (not semantic) bound.
AFFINE_ERROR_TOLERANCE = 1e-4

PREDICATE_TARGETS = [
    199.0,
    239.0,
    101.0,
    35.0,
    60.0,
    64.0,
    170.0,
    50.0,
    194.0,
    185.0,
    172.0,
    227.0,
    117.0,
    149.0,
    250.0,
    124.0,
]

CAPTURE_MODULES = [
    {
        "role": "h192",
        "module_name": "5437",
        "module_type": "torch.nn.modules.activation.ReLU",
        "shape": [192],
    },
    {
        "role": "a48",
        "module_name": "5438",
        "module_type": "torch.nn.modules.linear.Linear",
        "shape": [48],
    },
    {
        "role": "z48",
        "module_name": "5439",
        "module_type": "torch.nn.modules.activation.ReLU",
        "shape": [48],
    },
    {
        "role": "readout",
        "module_name": "5440",
        "module_type": "torch.nn.modules.linear.Linear",
        "shape": [1],
    },
    {
        "role": "output",
        "module_name": "5441",
        "module_type": "torch.nn.modules.activation.ReLU",
        "shape": [1],
    },
]
CAPTURE_SHAPES = {item["role"]: item["shape"] for item in CAPTURE_MODULES}


def float32_sha256(values: list[float]) -> str:
    digest = hashlib.sha256()
    for value in values:
        if type(value) not in {int, float} or not math.isfinite(value):
            raise ProbeSchemaError("Tensor snapshot values must be finite numbers.")
        try:
            encoded = struct.pack("<f", float(value))
        except (OverflowError, struct.error) as exc:
            raise ProbeSchemaError("Tensor snapshot value is not representable as float32.") from exc
        if struct.unpack("<f", encoded)[0] != float(value):
            raise ProbeSchemaError("Tensor snapshot value is not canonical float32.")
        digest.update(encoded)
    return digest.hexdigest()


def decode_q_values(h192: list[float]) -> list[float]:
    if len(h192) != 192:
        raise ProbeSchemaError("h192 must contain exactly 192 values.")
    return [
        sum((2**bit) * float(h192[8 * block + bit]) for bit in range(8))
        for block in range(24)
    ]


def predicate_values_from_a48(a48: list[float]) -> list[float]:
    if len(a48) != 48:
        raise ProbeSchemaError("a48 must contain exactly 48 values.")
    return [float(a48[16 + index]) + target for index, target in enumerate(PREDICATE_TARGETS)]


def predicate_values_from_q(q_values: list[float]) -> list[float]:
    if len(q_values) != 24:
        raise ProbeSchemaError("q_values must contain exactly 24 values.")
    return (
        [float(value) for value in q_values[0:4]]
        + [float(q_values[index]) - 2.0 * float(q_values[index + 4]) for index in range(4, 8)]
        + [float(value) for value in q_values[12:16]]
        + [
            float(q_values[index]) - 2.0 * float(q_values[index + 4])
            for index in range(16, 20)
        ]
    )


def indicator_values_from_z48(z48: list[float]) -> list[float]:
    if len(z48) != 48:
        raise ProbeSchemaError("z48 must contain exactly 48 values.")
    return [
        float(z48[index])
        - 2.0 * float(z48[16 + index])
        + float(z48[32 + index])
        for index in range(16)
    ]


def _md5_hex(payload: bytes) -> str:
    try:
        return hashlib.md5(payload, usedforsecurity=False).hexdigest()
    except TypeError:  # pragma: no cover - compatibility with older Python builds.
        return hashlib.md5(payload).hexdigest()


def decoded_predicate_hex(predicate_values: list[float]) -> str | None:
    decoded: list[int] = []
    for value in predicate_values:
        rounded = round(value)
        if not math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-6):
            return None
        if not 0 <= rounded <= 255:
            return None
        decoded.append(int(rounded))
    return bytes(decoded).hex()


def _codepoint_bytes(text: str) -> bytes | None:
    code_points = [ord(character) for character in text]
    if not all(0 <= value <= 255 for value in code_points):
        return None
    return bytes(code_points)


def _candidate_digest(payload: bytes | None) -> str | None:
    return _md5_hex(payload) if payload is not None else None


def digest_candidates(input_text: str, predicate_values: list[float]) -> dict[str, Any]:
    decoded = decoded_predicate_hex(predicate_values)
    truncated = input_text[:55]
    candidates = {
        "md5_utf8_hex": _md5_hex(input_text.encode("utf-8")),
        "md5_truncated_utf8_hex": _md5_hex(truncated.encode("utf-8")),
        "md5_latin1_hex": _candidate_digest(
            input_text.encode("latin-1") if all(ord(character) <= 255 for character in input_text) else None
        ),
        "md5_truncated_latin1_hex": _candidate_digest(
            truncated.encode("latin-1") if all(ord(character) <= 255 for character in truncated) else None
        ),
        "md5_codepoint_bytes_hex": _candidate_digest(_codepoint_bytes(input_text)),
        "md5_truncated_codepoint_bytes_hex": _candidate_digest(_codepoint_bytes(truncated)),
        "md5_padded_codepoint_bytes_hex": _candidate_digest(
            _codepoint_bytes(truncated.ljust(55, "\x00"))
        ),
    }
    return {
        "decoded_predicate_hex": decoded,
        **candidates,
        **{
            f"decoded_matches_{name.removeprefix('md5_').removesuffix('_hex')}": (
                value is not None and decoded == value
            )
            for name, value in candidates.items()
        },
    }


def derive_activation_values(
    input_text: str,
    tensors: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    h192 = tensors["h192"]["values"]
    a48 = tensors["a48"]["values"]
    z48 = tensors["z48"]["values"]
    readout = tensors["readout"]["values"]
    output = tensors["output"]["values"]
    q_values = decode_q_values(h192)
    predicate_values = predicate_values_from_q(q_values)
    a48_predicate_values = predicate_values_from_a48(a48)
    indicator_values = indicator_values_from_z48(z48)
    expected_a48 = (
        [value - (target + 1.0) for value, target in zip(predicate_values, PREDICATE_TARGETS)]
        + [value - target for value, target in zip(predicate_values, PREDICATE_TARGETS)]
        + [value - (target - 1.0) for value, target in zip(predicate_values, PREDICATE_TARGETS)]
    )
    expected_z48 = [max(value, 0.0) for value in a48]
    expected_readout = sum(indicator_values) - 15.0
    expected_output = max(float(readout[0]), 0.0)
    comparisons = (
        ("h192 and a48 predicate values", predicate_values, a48_predicate_values),
        ("a48", expected_a48, a48),
        ("z48", expected_z48, z48),
        ("readout", [expected_readout], readout),
        ("output", [expected_output], output),
    )
    for label, expected, actual in comparisons:
        if len(expected) != len(actual) or any(
            not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=AFFINE_ERROR_TOLERANCE)
            for left, right in zip(expected, actual)
        ):
            raise ProbeSchemaError(f"Captured {label} values are algebraically inconsistent.")
    matches = [
        math.isclose(value, target, rel_tol=0.0, abs_tol=1e-6)
        and math.isclose(indicator, 1.0, rel_tol=0.0, abs_tol=1e-6)
        for value, target, indicator in zip(
            predicate_values,
            PREDICATE_TARGETS,
            indicator_values,
        )
    ]
    return {
        "q_values": q_values,
        "predicate_values": predicate_values,
        "predicate_targets": list(PREDICATE_TARGETS),
        "indicator_values": indicator_values,
        "predicate_matches": matches,
        "matched_predicate_count": sum(matches),
        "digest_candidates": digest_candidates(input_text, predicate_values),
    }


def activation_signature(observation: dict[str, Any]) -> tuple[Any, ...]:
    tensors = observation["tensors"]
    return tuple(tensors[role]["float32_sha256"] for role in CAPTURE_SHAPES)


def summarize_activation_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    observation_count = 0
    deterministic_case_count = 0
    match_histogram: dict[int, int] = {}
    candidate_match_counts: dict[str, int] = {}
    for result in results:
        observations = result["observations"]
        observation_count += len(observations)
        deterministic_case_count += int(result["deterministic"])
        for observation in observations:
            derived = observation["derived"]
            matched = derived["matched_predicate_count"]
            match_histogram[matched] = match_histogram.get(matched, 0) + 1
            digests = derived["digest_candidates"]
            for key, value in digests.items():
                if key.startswith("decoded_matches_"):
                    candidate_match_counts[key] = candidate_match_counts.get(key, 0) + int(value)
    return {
        "case_count": len(results),
        "observation_count": observation_count,
        "deterministic_case_count": deterministic_case_count,
        "all_cases_deterministic": deterministic_case_count == len(results),
        "matched_predicate_count_histogram": [
            {"matched_predicate_count": count, "observation_count": occurrences}
            for count, occurrences in sorted(match_histogram.items())
        ],
        "candidate_match_counts": dict(sorted(candidate_match_counts.items())),
    }


def _validate_tensor_snapshot(
    snapshot: Any,
    *,
    role: str,
    expected_shape: list[int],
) -> None:
    if not isinstance(snapshot, dict) or set(snapshot) != {
        "dtype",
        "shape",
        "device",
        "values",
        "float32_sha256",
    }:
        raise ProbeSchemaError(f"Activation tensor {role!r} has an invalid field set.")
    if snapshot["dtype"] != "torch.float32":
        raise ProbeSchemaError(f"Activation tensor {role!r} must be torch.float32.")
    if snapshot["shape"] != expected_shape:
        raise ProbeSchemaError(f"Activation tensor {role!r} has an unexpected shape.")
    if snapshot["device"] != "cpu":
        raise ProbeSchemaError(f"Activation tensor {role!r} must be on the CPU.")
    values = snapshot["values"]
    expected_count = math.prod(expected_shape)
    if not isinstance(values, list) or len(values) != expected_count:
        raise ProbeSchemaError(f"Activation tensor {role!r} has an invalid value count.")
    expected_digest = float32_sha256(values)
    if snapshot["float32_sha256"] != expected_digest:
        raise ProbeSchemaError(f"Activation tensor {role!r} digest does not match its values.")


def _validate_observation(
    observation: Any,
    *,
    repetition: int,
    input_text: str,
) -> None:
    if not isinstance(observation, dict) or set(observation) != {
        "repetition",
        "tensors",
        "derived",
        "verification",
    }:
        raise ProbeSchemaError("Activation observation has an invalid field set.")
    if observation["repetition"] != repetition:
        raise ProbeSchemaError("Activation observation repetition is out of order.")
    tensors = observation["tensors"]
    if not isinstance(tensors, dict) or set(tensors) != set(CAPTURE_SHAPES):
        raise ProbeSchemaError("Activation observation tensor roles do not match.")
    for role, expected_shape in CAPTURE_SHAPES.items():
        _validate_tensor_snapshot(tensors[role], role=role, expected_shape=expected_shape)

    expected_derived = derive_activation_values(input_text, tensors)
    if observation["derived"] != expected_derived:
        raise ProbeSchemaError("Activation-derived values are inconsistent with the tensors.")

    verification = observation["verification"]
    expected_verification_keys = {
        "a48_max_abs_error",
        "z48_max_abs_error",
        "readout_max_abs_error",
        "output_max_abs_error",
        "returned_output_matches_hook",
    }
    if not isinstance(verification, dict) or set(verification) != expected_verification_keys:
        raise ProbeSchemaError("Activation verification fields are invalid.")
    for key in expected_verification_keys - {"returned_output_matches_hook"}:
        value = verification[key]
        if type(value) not in {int, float} or not math.isfinite(value):
            raise ProbeSchemaError(f"Activation verification field {key!r} is invalid.")
        if value < 0 or value > AFFINE_ERROR_TOLERANCE:
            raise ProbeSchemaError(f"Activation verification field {key!r} exceeds tolerance.")
    if verification["returned_output_matches_hook"] is not True:
        raise ProbeSchemaError("Returned model output does not match the final hook.")


def validate_activation_report(
    report: Any,
    *,
    expected_sha256: str,
    expected_manifest_sha256: str,
    manifest: dict[str, Any],
    repetitions: int,
) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise ProbeSchemaError("Activation report must be a JSON object.")
    required_keys = {
        "schema_version",
        "artifact",
        "probe_suite",
        "capture",
        "sandbox",
        "runtime",
        "execution",
        "results",
        "summary",
        "inference_executed",
    }
    if set(report) != required_keys:
        raise ProbeSchemaError("Activation report has an invalid top-level field set.")
    if (
        type(report["schema_version"]) is not int
        or report["schema_version"] != REPORT_SCHEMA_VERSION
    ):
        raise ProbeSchemaError("Unsupported activation report schema version.")

    artifact = report["artifact"]
    if not isinstance(artifact, dict) or set(artifact) != {
        "filename",
        "size_bytes",
        "sha256",
        "source_url",
    }:
        raise ProbeSchemaError("Activation report artifact metadata is invalid.")
    if artifact["sha256"] != expected_sha256:
        raise ProbeSchemaError("Activation report artifact hash does not match.")
    if not isinstance(artifact["filename"], str) or not artifact["filename"]:
        raise ProbeSchemaError("Activation report artifact filename is invalid.")
    if type(artifact["size_bytes"]) is not int or artifact["size_bytes"] <= 0:
        raise ProbeSchemaError("Activation report artifact size is invalid.")
    if not isinstance(artifact["source_url"], str) or not artifact["source_url"]:
        raise ProbeSchemaError("Activation report artifact source URL is invalid.")

    suite = report["probe_suite"]
    if not isinstance(suite, dict) or set(suite) != {
        "suite_id",
        "description",
        "manifest_filename",
        "manifest_sha256",
        "case_count",
        "repetitions",
    }:
        raise ProbeSchemaError("Activation report suite metadata is invalid.")
    expected_suite = {
        "suite_id": manifest["suite_id"],
        "description": manifest["description"],
        "manifest_sha256": expected_manifest_sha256,
        "case_count": len(manifest["cases"]),
        "repetitions": repetitions,
    }
    for key, expected in expected_suite.items():
        if suite.get(key) != expected:
            raise ProbeSchemaError(f"Activation report suite field {key!r} does not match.")

    capture = report["capture"]
    expected_capture = {
        "profile": CAPTURE_PROFILE,
        "scoped_hooks": True,
        "hooks_removed": True,
        "modules": CAPTURE_MODULES,
    }
    if capture != expected_capture:
        raise ProbeSchemaError("Activation capture metadata does not match.")

    expected_sandbox = {
        "profile": SANDBOX_PROFILE,
        "network_namespace": "isolated",
        "filesystem_policy": "Landlock allow-list",
        "model_access": "read-only",
        "manifest_access": "read-only",
        "host_credentials": "not allow-listed",
        "output_access": "staging-directory-only",
        "capabilities": "dropped",
        "no_new_privileges": True,
    }
    if report["sandbox"] != expected_sandbox:
        raise ProbeSchemaError("Activation report sandbox metadata does not match.")

    runtime = report["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != {
        "python",
        "torch",
        "cloudpickle",
        "platform",
    }:
        raise ProbeSchemaError("Activation report runtime metadata is invalid.")
    if not all(isinstance(value, str) and value for value in runtime.values()):
        raise ProbeSchemaError("Activation report runtime values are invalid.")

    expected_execution = {
        "inference_mode": True,
        "model_eval": True,
        "deterministic_algorithms": True,
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
        "seed": 0,
    }
    if report["execution"] != expected_execution:
        raise ProbeSchemaError("Activation report execution metadata does not match.")
    if report["inference_executed"] is not True:
        raise ProbeSchemaError("Activation report does not confirm inference execution.")

    results = report["results"]
    if not isinstance(results, list) or len(results) != len(manifest["cases"]):
        raise ProbeSchemaError("Activation result count does not match the manifest.")
    for case, result in zip(manifest["cases"], results):
        if not isinstance(result, dict) or set(result) != {
            "case",
            "input_length",
            "observations",
            "deterministic",
        }:
            raise ProbeSchemaError("Activation result has an invalid field set.")
        if result["case"] != case or result["input_length"] != len(case["input"]):
            raise ProbeSchemaError("Activation result does not match its manifest case.")
        observations = result["observations"]
        if not isinstance(observations, list) or len(observations) != repetitions:
            raise ProbeSchemaError("Activation result repetition count does not match.")
        for repetition, observation in enumerate(observations):
            _validate_observation(
                observation,
                repetition=repetition,
                input_text=case["input"],
            )
        deterministic = len({activation_signature(item) for item in observations}) == 1
        if result["deterministic"] is not deterministic:
            raise ProbeSchemaError("Activation determinism flag is inconsistent.")

    if report["summary"] != summarize_activation_results(results):
        raise ProbeSchemaError("Activation report summary is inconsistent.")
    return report
