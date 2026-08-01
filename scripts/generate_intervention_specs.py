from __future__ import annotations

import argparse
import json
import os
import struct
import tempfile
from pathlib import Path
from typing import Any

try:
    from scripts.activation_schema import float32_sha256
    from scripts.intervention_schema import SPEC_SCHEMA_VERSION, canonical_target_h192
    from scripts.probe_schema import load_probe_manifest, read_bounded_regular_file, sha256_bytes
except ModuleNotFoundError:  # Direct execution from scripts/.
    from activation_schema import float32_sha256  # type: ignore[no-redef]
    from intervention_schema import SPEC_SCHEMA_VERSION, canonical_target_h192  # type: ignore[no-redef]
    from probe_schema import load_probe_manifest, read_bounded_regular_file, sha256_bytes  # type: ignore[no-redef]


MAX_ANALYSIS_BYTES = 64 * 1024 * 1024


def _float32_values(values: list[float]) -> list[float]:
    """Round externally computed direction coefficients to canonical JSON float32 values."""
    return [struct.unpack("<f", struct.pack("<f", float(value)))[0] for value in values]


def _entry(
    identifier: str,
    mode: str,
    vector: list[float],
    strengths: list[float],
    case_ids: list[str],
    source: str,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "mode": mode,
        "vector": vector,
        "vector_float32_sha256": float32_sha256(vector),
        "strengths": strengths,
        "case_ids": case_ids,
        "source": source,
    }


def _break_vector(predicate_index: int) -> list[float]:
    target = canonical_target_h192()
    byte_block = predicate_index if predicate_index < 8 else predicate_index + 4
    start = byte_block * 8
    # Change exactly one recovered MD5 predicate byte, retaining a canonical bit vector.
    target[start] = 1.0 - target[start]
    return target


def build_canonical_spec(
    manifest: dict[str, Any], *, source_report_sha256: str, manifest_sha256: str,
    smoke_case_id: str | None = None,
) -> dict[str, Any]:
    case_ids = [case["id"] for case in manifest["cases"]]
    if smoke_case_id is not None:
        if smoke_case_id not in case_ids:
            raise ValueError("Canonical smoke case must be a manifest case id.")
        case_ids = [smoke_case_id]
    interventions = [
        _entry(
            "replace-canonical-all-predicates", "replace", canonical_target_h192(), [1.0], case_ids,
            "Recovered 24-byte h192 predicate layout; sets all 16 equality predicates true.",
        )
    ]
    interventions.extend(
        _entry(
            f"replace-break-predicate-{index:02d}", "replace", _break_vector(index), [1.0], case_ids,
            f"Canonical h192 layout with recovered predicate {index} deliberately mismatched.",
        )
        for index in range(16)
    )
    return {
        "schema_version": SPEC_SCHEMA_VERSION,
        "suite_id": "m4-canonical-gate-v1",
        "description": (
            "Hash-bound final-circuit positive and negative controls. Replacement vectors are "
            "canonical h192 bit representations derived from the recovered 16 MD5 predicates."
        ),
        "source_activation_report_sha256": source_report_sha256,
        "source_manifest_sha256": manifest_sha256,
        "interventions": interventions,
    }


def build_semantic_spec(
    manifest: dict[str, Any],
    analysis: dict[str, Any],
    *,
    source_report_sha256: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    if analysis.get("analysis_kind") != "held_out_semantic_directions":
        raise ValueError("Semantic intervention generation requires a semantic direction analysis.")
    if analysis.get("source", {}).get("activation_report_sha256") != source_report_sha256:
        raise ValueError("Semantic analysis is not bound to the supplied activation report.")
    case_by_id = {case["id"]: case for case in manifest["cases"]}
    interventions = []
    for slot in ("left", "right"):
        for fold in analysis["slots"][slot]["folds"]:
            predictions = fold["evaluation"]["predictions"]
            for category, vector in fold["directions"].items():
                selected_case_ids = [
                    item["case_id"] for item in predictions
                    if item["actual"] == category and item["case_id"] in case_by_id
                ]
                if not selected_case_ids:
                    raise ValueError("Semantic fold did not retain category-held-out cases.")
                interventions.append(
                    _entry(
                        f"add-{slot}-fold-{fold['fold_index']}-{category}", "add",
                        _float32_values([float(value) for value in vector]), [-1.0, 0.0, 1.0],
                        selected_case_ids,
                        (
                            f"Held-out {slot} one-vs-rest direction for category {category}; "
                            f"fold {fold['fold_index']} with whole-lexeme holdout."
                        ),
                    )
                )
    return {
        "schema_version": SPEC_SCHEMA_VERSION,
        "suite_id": "m4-held-out-semantic-directions-v1",
        "description": (
            "All leave-one-lexeme-per-category-out h192 semantic directions, with negative, "
            "zero, and positive additive strengths on their held-out category cases."
        ),
        "source_activation_report_sha256": source_report_sha256,
        "source_manifest_sha256": manifest_sha256,
        "interventions": interventions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate hash-bound Milestone 4 intervention specs.")
    parser.add_argument("--kind", choices=("canonical", "semantic"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-activation-report", type=Path, required=True)
    parser.add_argument("--semantic-analysis", type=Path)
    parser.add_argument(
        "--canonical-smoke-case",
        help="Generate only the positive control and predicate-00 negative control for one case.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest, _payload, manifest_sha256 = load_probe_manifest(args.manifest)
        source_sha256 = sha256_bytes(read_bounded_regular_file(args.source_activation_report, MAX_ANALYSIS_BYTES))
        if args.kind == "canonical":
            spec = build_canonical_spec(
                manifest, source_report_sha256=source_sha256, manifest_sha256=manifest_sha256,
                smoke_case_id=args.canonical_smoke_case,
            )
            if args.canonical_smoke_case is not None:
                spec["suite_id"] = "m4-canonical-gate-smoke-v1"
                spec["description"] = (
                    "Fast sandbox smoke subset of the canonical positive and predicate-00 "
                    "negative controls. Generate without --canonical-smoke-case for full coverage."
                )
                spec["interventions"] = spec["interventions"][:2]
        else:
            if args.semantic_analysis is None:
                raise ValueError("--semantic-analysis is required for semantic intervention specs.")
            analysis = json.loads(read_bounded_regular_file(args.semantic_analysis, MAX_ANALYSIS_BYTES))
            spec = build_semantic_spec(
                manifest, analysis, source_report_sha256=source_sha256,
                manifest_sha256=manifest_sha256,
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Intervention spec generation failed: {exc}")
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=args.output.parent,
                                     prefix=f".{args.output.name}.", suffix=".tmp", delete=False) as handle:
        temporary_path = Path(handle.name)
        json.dump(spec, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary_path, args.output)
    print(f"Intervention spec written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
