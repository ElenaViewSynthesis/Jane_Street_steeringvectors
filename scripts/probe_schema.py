from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_CASES = 512
MAX_INPUT_CHARACTERS = 4_096
MAX_TOTAL_INPUT_CHARACTERS = 256 * 1024
MAX_FACTORS_PER_CASE = 32
MAX_DESCRIPTION_CHARACTERS = 2_000
MAX_NOTES_CHARACTERS = 2_000

IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
FACTOR_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class ProbeSchemaError(ValueError):
    """Raised when a probe manifest or report violates its data contract."""


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProbeSchemaError(f"JSON object contains duplicate key {key!r}.")
        result[key] = value
    return result


def loads_json(payload: bytes | str) -> Any:
    try:
        return json.loads(payload, object_pairs_hook=_object_without_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbeSchemaError(f"Invalid JSON: {exc}") from exc


def read_bounded_regular_file(path: Path, maximum_bytes: int) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ProbeSchemaError(f"Unable to inspect {path}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ProbeSchemaError(f"{path} must be a single regular file, not a link.")
    if metadata.st_size > maximum_bytes:
        raise ProbeSchemaError(f"{path} exceeds the {maximum_bytes}-byte size limit.")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            payload = handle.read(maximum_bytes + 1)
    except OSError as exc:
        raise ProbeSchemaError(f"Unable to read {path}: {exc}") from exc
    if len(payload) > maximum_bytes:
        raise ProbeSchemaError(f"{path} exceeds the {maximum_bytes}-byte size limit.")
    return payload


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER_PATTERN.fullmatch(value):
        raise ProbeSchemaError(
            f"{label} must match {IDENTIFIER_PATTERN.pattern!r}."
        )
    return value


def _validate_factors(factors: Any, case_id: str) -> None:
    if not isinstance(factors, dict):
        raise ProbeSchemaError(f"Case {case_id!r} factors must be a JSON object.")
    if len(factors) > MAX_FACTORS_PER_CASE:
        raise ProbeSchemaError(
            f"Case {case_id!r} exceeds the {MAX_FACTORS_PER_CASE}-factor limit."
        )
    for name, value in factors.items():
        if not isinstance(name, str) or not FACTOR_NAME_PATTERN.fullmatch(name):
            raise ProbeSchemaError(
                f"Case {case_id!r} has invalid factor name {name!r}."
            )
        if value is not None and type(value) not in {bool, int, str}:
            raise ProbeSchemaError(
                f"Case {case_id!r} factor {name!r} must be a string, integer, "
                "boolean, or null."
            )


def validate_probe_manifest(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ProbeSchemaError("Probe manifest must be a JSON object.")
    required_keys = {"schema_version", "suite_id", "description", "cases"}
    unknown_keys = set(document) - required_keys
    missing_keys = required_keys - set(document)
    if unknown_keys:
        raise ProbeSchemaError(
            f"Probe manifest contains unknown keys: {', '.join(sorted(unknown_keys))}."
        )
    if missing_keys:
        raise ProbeSchemaError(
            f"Probe manifest is missing keys: {', '.join(sorted(missing_keys))}."
        )
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != MANIFEST_SCHEMA_VERSION
    ):
        raise ProbeSchemaError("Unsupported probe manifest schema version.")
    _require_identifier(document["suite_id"], "suite_id")

    description = document["description"]
    if not isinstance(description, str) or not description.strip():
        raise ProbeSchemaError("Probe manifest description must be a non-empty string.")
    if len(description) > MAX_DESCRIPTION_CHARACTERS:
        raise ProbeSchemaError("Probe manifest description is too long.")

    cases = document["cases"]
    if not isinstance(cases, list) or not cases:
        raise ProbeSchemaError("Probe manifest cases must be a non-empty array.")
    if len(cases) > MAX_CASES:
        raise ProbeSchemaError(f"Probe manifest exceeds the {MAX_CASES}-case limit.")

    seen_ids: set[str] = set()
    total_input_characters = 0
    allowed_case_keys = {"id", "input", "family", "factors", "notes"}
    required_case_keys = {"id", "input", "family", "factors"}
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ProbeSchemaError(f"Case at index {index} must be a JSON object.")
        unknown_case_keys = set(case) - allowed_case_keys
        missing_case_keys = required_case_keys - set(case)
        if unknown_case_keys:
            raise ProbeSchemaError(
                f"Case at index {index} contains unknown keys: "
                f"{', '.join(sorted(unknown_case_keys))}."
            )
        if missing_case_keys:
            raise ProbeSchemaError(
                f"Case at index {index} is missing keys: "
                f"{', '.join(sorted(missing_case_keys))}."
            )

        case_id = _require_identifier(case["id"], f"case[{index}].id")
        if case_id in seen_ids:
            raise ProbeSchemaError(f"Duplicate probe case id {case_id!r}.")
        seen_ids.add(case_id)
        _require_identifier(case["family"], f"case[{index}].family")

        input_text = case["input"]
        if not isinstance(input_text, str):
            raise ProbeSchemaError(f"Case {case_id!r} input must be a string.")
        if len(input_text) > MAX_INPUT_CHARACTERS:
            raise ProbeSchemaError(
                f"Case {case_id!r} exceeds the {MAX_INPUT_CHARACTERS}-character limit."
            )
        total_input_characters += len(input_text)
        _validate_factors(case["factors"], case_id)

        notes = case.get("notes")
        if notes is not None and (
            not isinstance(notes, str) or len(notes) > MAX_NOTES_CHARACTERS
        ):
            raise ProbeSchemaError(
                f"Case {case_id!r} notes must be a string no longer than "
                f"{MAX_NOTES_CHARACTERS} characters."
            )

    if total_input_characters > MAX_TOTAL_INPUT_CHARACTERS:
        raise ProbeSchemaError(
            "Probe manifest exceeds the total input-character budget."
        )
    return document


def load_probe_manifest(path: Path) -> tuple[dict[str, Any], bytes, str]:
    payload = read_bounded_regular_file(path, MAX_MANIFEST_BYTES)
    manifest = validate_probe_manifest(loads_json(payload))
    return manifest, payload, sha256_bytes(payload)


def summarize_probe_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    output_counts: dict[tuple[str, float], int] = {}
    observation_count = 0
    deterministic_case_count = 0
    for result in results:
        observations = result["observations"]
        observation_count += len(observations)
        if result["deterministic"]:
            deterministic_case_count += 1
        for observation in observations:
            key = (observation["float_hex"], observation["value"])
            output_counts[key] = output_counts.get(key, 0) + 1

    rendered_counts = [
        {"float_hex": float_hex, "value": value, "count": count}
        for (float_hex, value), count in sorted(output_counts.items())
    ]
    return {
        "case_count": len(results),
        "observation_count": observation_count,
        "deterministic_case_count": deterministic_case_count,
        "all_cases_deterministic": deterministic_case_count == len(results),
        "unique_output_count": len(output_counts),
        "output_counts": rendered_counts,
    }
