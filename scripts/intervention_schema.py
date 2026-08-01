from __future__ import annotations

import math
from typing import Any

try:
    from scripts.activation_schema import (
        AFFINE_ERROR_TOLERANCE,
        CAPTURE_SHAPES,
        PREDICATE_TARGETS,
        _validate_observation,
        activation_signature,
        decode_q_values,
        float32_sha256,
        predicate_values_from_q,
    )
    from scripts.probe_schema import ProbeSchemaError, loads_json, sha256_bytes
except ModuleNotFoundError:  # Direct execution from scripts/.
    from activation_schema import (  # type: ignore[no-redef]
        AFFINE_ERROR_TOLERANCE,
        CAPTURE_SHAPES,
        PREDICATE_TARGETS,
        _validate_observation,
        activation_signature,
        decode_q_values,
        float32_sha256,
        predicate_values_from_q,
    )
    from probe_schema import ProbeSchemaError, loads_json, sha256_bytes  # type: ignore[no-redef]


SPEC_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
MAX_INTERVENTIONS = 64
MAX_STRENGTHS = 9
MAX_ABS_STRENGTH = 32.0


def canonical_target_h192() -> list[float]:
    bytes_by_block = (
        [int(value) for value in PREDICATE_TARGETS[0:4]]
        + [int(value) for value in PREDICATE_TARGETS[4:8]]
        + [0] * 4
        + [int(value) for value in PREDICATE_TARGETS[8:12]]
        + [int(value) for value in PREDICATE_TARGETS[12:16]]
        + [0] * 4
    )
    return [
        float((byte >> bit) & 1)
        for byte in bytes_by_block
        for bit in range(8)
    ]


def _validate_vector(vector: Any, label: str) -> list[float]:
    if not isinstance(vector, list) or len(vector) != 192:
        raise ProbeSchemaError(f"{label} must contain exactly 192 values.")
    values = []
    for value in vector:
        if type(value) not in {int, float} or not math.isfinite(value):
            raise ProbeSchemaError(f"{label} must contain finite numeric values.")
        values.append(float(value))
    try:
        float32_sha256(values)
    except ProbeSchemaError as exc:
        raise ProbeSchemaError(f"{label} must contain canonical float32 values.") from exc
    return values


def validate_intervention_spec(document: Any, *, case_ids: set[str]) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ProbeSchemaError("Intervention spec must be a JSON object.")
    required = {
        "schema_version",
        "suite_id",
        "description",
        "source_activation_report_sha256",
        "source_manifest_sha256",
        "interventions",
    }
    if set(document) != required:
        raise ProbeSchemaError("Intervention spec has an invalid top-level field set.")
    if type(document["schema_version"]) is not int or document["schema_version"] != SPEC_SCHEMA_VERSION:
        raise ProbeSchemaError("Unsupported intervention spec schema version.")
    if not isinstance(document["suite_id"], str) or not document["suite_id"]:
        raise ProbeSchemaError("Intervention suite_id is invalid.")
    if not isinstance(document["description"], str) or not document["description"].strip():
        raise ProbeSchemaError("Intervention description is invalid.")
    for key in ("source_activation_report_sha256", "source_manifest_sha256"):
        value = document[key]
        if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ProbeSchemaError(f"Intervention {key} is invalid.")
    interventions = document["interventions"]
    if not isinstance(interventions, list) or not interventions or len(interventions) > MAX_INTERVENTIONS:
        raise ProbeSchemaError("Intervention spec has an invalid intervention count.")
    seen_ids: set[str] = set()
    for item in interventions:
        required_item = {
            "id",
            "mode",
            "vector",
            "vector_float32_sha256",
            "strengths",
            "case_ids",
            "source",
        }
        if not isinstance(item, dict) or set(item) != required_item:
            raise ProbeSchemaError("Intervention entry has an invalid field set.")
        identifier = item["id"]
        if not isinstance(identifier, str) or not identifier or identifier in seen_ids:
            raise ProbeSchemaError("Intervention id is invalid or duplicated.")
        seen_ids.add(identifier)
        if item["mode"] not in {"add", "replace"}:
            raise ProbeSchemaError("Intervention mode must be add or replace.")
        vector = _validate_vector(item["vector"], f"Intervention {identifier!r} vector")
        if item["vector_float32_sha256"] != float32_sha256(vector):
            raise ProbeSchemaError("Intervention vector hash does not match its values.")
        strengths = item["strengths"]
        if not isinstance(strengths, list) or not strengths or len(strengths) > MAX_STRENGTHS:
            raise ProbeSchemaError("Intervention strengths are invalid.")
        if len({float(value) for value in strengths if type(value) in {int, float}}) != len(strengths):
            raise ProbeSchemaError("Intervention strengths must be unique.")
        for value in strengths:
            if type(value) not in {int, float} or not math.isfinite(value):
                raise ProbeSchemaError("Intervention strength must be finite and numeric.")
            if abs(float(value)) > MAX_ABS_STRENGTH:
                raise ProbeSchemaError("Intervention strength exceeds the safety bound.")
        if item["mode"] == "replace" and any(
            not 0.0 <= float(value) <= 1.0 for value in strengths
        ):
            raise ProbeSchemaError("Replacement strengths must be in the closed interval [0, 1].")
        selected_cases = item["case_ids"]
        if not isinstance(selected_cases, list) or not selected_cases:
            raise ProbeSchemaError("Intervention case_ids are invalid.")
        if len(selected_cases) != len(set(selected_cases)) or any(
            not isinstance(case_id, str) or case_id not in case_ids for case_id in selected_cases
        ):
            raise ProbeSchemaError("Intervention case_ids must be unique manifest case ids.")
        if not isinstance(item["source"], str) or not item["source"]:
            raise ProbeSchemaError("Intervention source is invalid.")
    return document


def load_intervention_spec(payload: bytes, *, case_ids: set[str]) -> tuple[dict[str, Any], str]:
    spec = validate_intervention_spec(loads_json(payload), case_ids=case_ids)
    return spec, sha256_bytes(payload)


def _predicted_tail(predicate_values: list[float]) -> tuple[list[float], float, float]:
    a48 = (
        [value - (target + 1.0) for value, target in zip(predicate_values, PREDICATE_TARGETS)]
        + [value - target for value, target in zip(predicate_values, PREDICATE_TARGETS)]
        + [value - (target - 1.0) for value, target in zip(predicate_values, PREDICATE_TARGETS)]
    )
    z48 = [max(value, 0.0) for value in a48]
    indicators = [
        z48[index] - 2.0 * z48[16 + index] + z48[32 + index]
        for index in range(16)
    ]
    readout = sum(indicators) - 15.0
    return indicators, readout, max(readout, 0.0)


def derive_intervention_comparison(
    baseline: dict[str, Any],
    observed: dict[str, Any],
) -> dict[str, Any]:
    baseline_h = baseline["tensors"]["h192"]["values"]
    observed_h = observed["tensors"]["h192"]["values"]
    delta_h = [after - before for before, after in zip(baseline_h, observed_h)]
    predicted_delta = predicate_values_from_q(decode_q_values(delta_h))
    baseline_predicates = baseline["derived"]["predicate_values"]
    observed_predicates = observed["derived"]["predicate_values"]
    actual_delta = [
        after - before for before, after in zip(baseline_predicates, observed_predicates)
    ]
    predicted_predicates = [
        value + delta for value, delta in zip(baseline_predicates, predicted_delta)
    ]
    _indicators, predicted_readout, predicted_output = _predicted_tail(predicted_predicates)
    baseline_readout = baseline["tensors"]["readout"]["values"][0]
    observed_readout = observed["tensors"]["readout"]["values"][0]
    baseline_output = baseline["tensors"]["output"]["values"][0]
    observed_output = observed["tensors"]["output"]["values"][0]
    changed_relu_rows = [
        index
        for index, (before, after) in enumerate(
            zip(baseline["tensors"]["a48"]["values"], observed["tensors"]["a48"]["values"])
        )
        if (before > 0.0) != (after > 0.0)
    ]
    return {
        "predicted_predicate_delta": predicted_delta,
        "actual_predicate_delta": actual_delta,
        "predicate_delta_max_abs_error": max(
            abs(left - right) for left, right in zip(predicted_delta, actual_delta)
        ),
        "predicted_readout_delta": predicted_readout - baseline_readout,
        "actual_readout_delta": observed_readout - baseline_readout,
        "readout_delta_abs_error": abs(predicted_readout - observed_readout),
        "predicted_output_delta": predicted_output - baseline_output,
        "actual_output_delta": observed_output - baseline_output,
        "output_delta_abs_error": abs(predicted_output - observed_output),
        "predicate_relu_crossing_rows": changed_relu_rows,
        "final_relu_crossed": (baseline_readout > 0.0) != (observed_readout > 0.0),
    }


def validate_intervention_report(
    report: Any,
    *,
    manifest: dict[str, Any],
    expected_model_sha256: str,
    expected_spec_sha256: str,
    spec: dict[str, Any],
    expected_source_report_sha256: str,
    repetitions: int,
) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise ProbeSchemaError("Intervention report must be a JSON object.")
    required = {
        "schema_version", "artifact", "probe_suite", "source_activation_report",
        "intervention_spec", "execution", "results", "summary", "inference_executed",
    }
    if set(report) != required or report["schema_version"] != REPORT_SCHEMA_VERSION:
        raise ProbeSchemaError("Intervention report has an invalid top-level field set.")
    if report["artifact"].get("sha256") != expected_model_sha256:
        raise ProbeSchemaError("Intervention report artifact hash does not match.")
    if report["source_activation_report"] != {"sha256": expected_source_report_sha256}:
        raise ProbeSchemaError("Intervention source activation report is not bound.")
    if report["intervention_spec"] != {
        "suite_id": spec["suite_id"],
        "sha256": expected_spec_sha256,
    }:
        raise ProbeSchemaError("Intervention spec is not bound.")
    if report["probe_suite"] != {
        "suite_id": manifest["suite_id"],
        "manifest_sha256": spec["source_manifest_sha256"],
        "repetitions": repetitions,
    }:
        raise ProbeSchemaError("Intervention report suite metadata is invalid.")
    if report["execution"] != {"inference_mode": True, "model_eval": True, "hooks_removed": True}:
        raise ProbeSchemaError("Intervention execution metadata is invalid.")
    if report["inference_executed"] is not True:
        raise ProbeSchemaError("Intervention report does not confirm inference execution.")
    spec_by_id = {item["id"]: item for item in spec["interventions"]}
    if not isinstance(report["results"], list) or len(report["results"]) != len(manifest["cases"]):
        raise ProbeSchemaError("Intervention report case count is invalid.")
    observation_count = 0
    for case, result in zip(manifest["cases"], report["results"]):
        if not isinstance(result, dict) or set(result) != {"case", "baseline", "interventions"}:
            raise ProbeSchemaError("Intervention result has an invalid field set.")
        if result["case"] != case:
            raise ProbeSchemaError("Intervention result case is not manifest-bound.")
        baseline = result["baseline"]
        if not isinstance(baseline, list) or len(baseline) != repetitions:
            raise ProbeSchemaError("Intervention baseline repetitions are invalid.")
        for repetition, observation in enumerate(baseline):
            _validate_observation(observation, repetition=repetition, input_text=case["input"])
        expected_ids = {
            item["id"] for item in spec["interventions"] if case["id"] in item["case_ids"]
        }
        actual_ids = {
            item.get("intervention_id")
            for item in result["interventions"]
            if isinstance(item, dict)
        }
        if actual_ids != expected_ids:
            raise ProbeSchemaError("Intervention result coverage does not match the spec.")
        for group in result["interventions"]:
            if not isinstance(group, dict) or set(group) != {
                "intervention_id", "strength", "observations", "deterministic"
            }:
                raise ProbeSchemaError("Intervention group has an invalid field set.")
            spec_item = spec_by_id.get(group["intervention_id"])
            if spec_item is None or group["strength"] not in spec_item["strengths"]:
                raise ProbeSchemaError("Intervention group is not spec-bound.")
            observations = group["observations"]
            if not isinstance(observations, list) or len(observations) != repetitions:
                raise ProbeSchemaError("Intervention observation repetitions are invalid.")
            signatures = []
            for repetition, item in enumerate(observations):
                if not isinstance(item, dict) or set(item) != {"activation", "comparison"}:
                    raise ProbeSchemaError("Intervention observation has an invalid field set.")
                _validate_observation(item["activation"], repetition=repetition, input_text=case["input"])
                expected_comparison = derive_intervention_comparison(baseline[repetition], item["activation"])
                if item["comparison"] != expected_comparison:
                    raise ProbeSchemaError("Intervention comparison is inconsistent.")
                for key in (
                    "predicate_delta_max_abs_error", "readout_delta_abs_error", "output_delta_abs_error"
                ):
                    if item["comparison"][key] > AFFINE_ERROR_TOLERANCE:
                        raise ProbeSchemaError("Intervention analytic prediction exceeds tolerance.")
                signatures.append(activation_signature(item["activation"]))
                observation_count += 1
            if group["deterministic"] is not (len(set(signatures)) == 1):
                raise ProbeSchemaError("Intervention determinism flag is inconsistent.")
    expected_summary = {"intervention_observation_count": observation_count}
    if report["summary"] != expected_summary:
        raise ProbeSchemaError("Intervention summary is inconsistent.")
    return report
