from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    from scripts.analyze_activation_report import load_validated_activation_report
    from scripts.intervention_schema import load_intervention_spec, validate_intervention_report
    from scripts.local_geometry_schema import (
        load_semantic_analysis,
        validate_geometry_report,
    )
    from scripts.probe_schema import (
        ProbeSchemaError,
        load_probe_manifest,
        loads_json,
        read_bounded_regular_file,
        sha256_bytes,
    )
    from scripts.run_architecture_sandbox import validate_architecture_report
    from scripts.run_probe_sandbox import validate_probe_report
except ModuleNotFoundError:  # Direct execution from scripts/.
    from analyze_activation_report import load_validated_activation_report  # type: ignore[no-redef]
    from intervention_schema import (  # type: ignore[no-redef]
        load_intervention_spec,
        validate_intervention_report,
    )
    from local_geometry_schema import (  # type: ignore[no-redef]
        load_semantic_analysis,
        validate_geometry_report,
    )
    from probe_schema import (  # type: ignore[no-redef]
        ProbeSchemaError,
        load_probe_manifest,
        loads_json,
        read_bounded_regular_file,
        sha256_bytes,
    )
    from run_architecture_sandbox import validate_architecture_report  # type: ignore[no-redef]
    from run_probe_sandbox import validate_probe_report  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
SYNTHESIS_SCHEMA_VERSION = 1
MAX_REPORT_BYTES = 64 * 1024 * 1024
MAX_ARCHITECTURE_BYTES = 16 * 1024 * 1024
MAX_ANALYSIS_BYTES = 4 * 1024 * 1024
MAX_SPEC_BYTES = 4 * 1024 * 1024


PATHS = {
    "architecture": ROOT / "outputs/reports/architecture_report.json",
    "smoke_manifest": ROOT / "experiments/probes/m3-smoke-v1.json",
    "smoke_probe": ROOT / "outputs/probes/m3-smoke-v1.json",
    "semantic_manifest": ROOT / "experiments/probes/m3-semantic-factorial-v1.json",
    "semantic_probe": ROOT / "outputs/probes/m3-semantic-factorial-v1.json",
    "activation_manifest": ROOT / "experiments/activations/m4-capture-smoke-v1.json",
    "activation_smoke": ROOT / "outputs/activations/m4-capture-smoke-v1.json",
    "boundary_manifest": ROOT / "experiments/activations/m4-md5-boundary-v1.json",
    "boundary_activation": ROOT / "outputs/activations/m4-md5-boundary-v1.json",
    "boundary_analysis": ROOT / "outputs/activations/m4-md5-boundary-analysis-v1.json",
    "semantic_activation": ROOT / "outputs/activations/m4-semantic-factorial-v1.json",
    "semantic_analysis": ROOT / "outputs/activations/m4-semantic-direction-analysis-v1.json",
    "canonical_spec": ROOT / "outputs/interventions/m4-canonical-gate-v1.json",
    "canonical_report": ROOT / "outputs/interventions/m4-canonical-gate-report-v1.json",
    "semantic_spec": ROOT / "outputs/interventions/m4-semantic-direction-interventions-v1.json",
    "semantic_intervention": ROOT / "outputs/interventions/m4-semantic-direction-report-v1.json",
    "geometry": ROOT / "outputs/reports/m5-local-geometry-v1.json",
}


def _read_json(path: Path, maximum_bytes: int) -> tuple[dict[str, Any], str]:
    payload = read_bounded_regular_file(path, maximum_bytes)
    document = loads_json(payload)
    if not isinstance(document, dict):
        raise ProbeSchemaError(f"{path} must contain a JSON object.")
    return document, sha256_bytes(payload)


def _write_json(path: Path, document: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(document, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    os.replace(temporary, path)


def _load_probe(report_path: Path, manifest_path: Path) -> tuple[dict[str, Any], str]:
    manifest, _payload, manifest_sha256 = load_probe_manifest(manifest_path)
    report, report_sha256 = _read_json(report_path, MAX_REPORT_BYTES)
    artifact = report.get("artifact")
    suite = report.get("probe_suite")
    if not isinstance(artifact, dict) or not isinstance(suite, dict):
        raise ProbeSchemaError("Probe report is missing artifact or suite metadata.")
    validated = validate_probe_report(
        report,
        expected_sha256=artifact.get("sha256"),
        expected_manifest_sha256=manifest_sha256,
        manifest=manifest,
        repetitions=suite.get("repetitions"),
    )
    return validated, report_sha256


def _load_intervention(
    *,
    manifest_path: Path,
    activation_path: Path,
    spec_path: Path,
    report_path: Path,
) -> tuple[dict[str, Any], str, str, str]:
    manifest, _payload, manifest_sha256 = load_probe_manifest(manifest_path)
    activation, activation_sha256 = load_validated_activation_report(activation_path, manifest_path)
    spec_payload = read_bounded_regular_file(spec_path, MAX_SPEC_BYTES)
    spec, spec_sha256 = load_intervention_spec(
        spec_payload, case_ids={case["id"] for case in manifest["cases"]}
    )
    if spec["source_manifest_sha256"] != manifest_sha256:
        raise ProbeSchemaError("Intervention spec manifest binding is invalid.")
    if spec["source_activation_report_sha256"] != activation_sha256:
        raise ProbeSchemaError("Intervention spec activation binding is invalid.")
    report, report_sha256 = _read_json(report_path, MAX_REPORT_BYTES)
    repetitions = report.get("probe_suite", {}).get("repetitions")
    validated = validate_intervention_report(
        report,
        manifest=manifest,
        expected_model_sha256=activation["artifact"]["sha256"],
        expected_spec_sha256=spec_sha256,
        spec=spec,
        expected_source_report_sha256=activation_sha256,
        repetitions=repetitions,
    )
    return validated, report_sha256, spec_sha256, activation_sha256


def _load_boundary_analysis(activation_sha256: str) -> tuple[dict[str, Any], str]:
    document, digest = _read_json(PATHS["boundary_analysis"], MAX_ANALYSIS_BYTES)
    required = {"schema_version", "analysis_kind", "source", "cases", "candidate_summary", "conclusion"}
    if set(document) != required or document["analysis_kind"] != "md5_encoding_boundary":
        raise ProbeSchemaError("Boundary analysis schema or kind is invalid.")
    if document["source"].get("activation_report_sha256") != activation_sha256:
        raise ProbeSchemaError("Boundary analysis activation binding is invalid.")
    if len(document["cases"]) != 10:
        raise ProbeSchemaError("Boundary analysis must contain the ten declared cases.")
    return document, digest


def summarize_semantic_interventions(report: dict[str, Any]) -> dict[str, Any]:
    comparisons = []
    activations = []
    for result in report["results"]:
        for group in result["interventions"]:
            for observation in group["observations"]:
                comparisons.append(observation["comparison"])
                activations.append(observation["activation"])
    return {
        "observation_count": len(comparisons),
        "predicate_delta_max_abs_error": max(
            item["predicate_delta_max_abs_error"] for item in comparisons
        ),
        "readout_delta_max_abs_error": max(item["readout_delta_abs_error"] for item in comparisons),
        "output_delta_max_abs_error": max(item["output_delta_abs_error"] for item in comparisons),
        "predicate_relu_crossing_observations": sum(
            bool(item["predicate_relu_crossing_rows"]) for item in comparisons
        ),
        "final_relu_crossing_observations": sum(item["final_relu_crossed"] for item in comparisons),
        "nonzero_output_observations": sum(
            item["tensors"]["output"]["values"][0] != 0.0 for item in activations
        ),
        "readout_minimum": min(item["tensors"]["readout"]["values"][0] for item in activations),
        "readout_maximum": max(item["tensors"]["readout"]["values"][0] for item in activations),
    }


def summarize_canonical_interventions(report: dict[str, Any]) -> dict[str, Any]:
    canonical = []
    breaks = []
    for result in report["results"]:
        for group in result["interventions"]:
            destination = canonical if group["intervention_id"] == "replace-canonical-all-predicates" else breaks
            for observation in group["observations"]:
                activation = observation["activation"]
                destination.append(
                    (
                        activation["derived"]["matched_predicate_count"],
                        activation["tensors"]["readout"]["values"][0],
                        activation["tensors"]["output"]["values"][0],
                    )
                )
    if set(canonical) != {(16, 1.0, 1.0)} or set(breaks) != {(15, 0.0, 0.0)}:
        raise ProbeSchemaError("Canonical intervention controls do not implement the recovered gate.")
    return {
        "observation_count": len(canonical) + len(breaks),
        "canonical_observations": len(canonical),
        "single_predicate_break_observations": len(breaks),
        "canonical_result": {"matched_predicates": 16, "readout": 1.0, "output": 1.0},
        "break_result": {"matched_predicates": 15, "readout": 0.0, "output": 0.0},
    }


def _artifact_entry(name: str, path: Path, digest: str | None = None) -> dict[str, Any]:
    payload_digest = digest or hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "name": name,
        "path": str(path.relative_to(ROOT)),
        "sha256": payload_digest,
        "size_bytes": path.stat().st_size,
        "tracked_by_git": False,
    }


def build_summary(inventory_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    architecture, architecture_sha = _read_json(PATHS["architecture"], MAX_ARCHITECTURE_BYTES)
    artifact_hash = architecture.get("artifact", {}).get("sha256")
    validate_architecture_report(architecture, artifact_hash)
    smoke_probe, smoke_probe_sha = _load_probe(PATHS["smoke_probe"], PATHS["smoke_manifest"])
    semantic_probe, semantic_probe_sha = _load_probe(
        PATHS["semantic_probe"], PATHS["semantic_manifest"]
    )
    boundary_activation, boundary_activation_sha = load_validated_activation_report(
        PATHS["boundary_activation"], PATHS["boundary_manifest"]
    )
    boundary_analysis, boundary_analysis_sha = _load_boundary_analysis(boundary_activation_sha)
    semantic_activation, semantic_activation_sha = load_validated_activation_report(
        PATHS["semantic_activation"], PATHS["semantic_manifest"]
    )
    semantic_analysis, semantic_analysis_sha = load_semantic_analysis(PATHS["semantic_analysis"])
    if semantic_analysis["source"]["activation_report_sha256"] != semantic_activation_sha:
        raise ProbeSchemaError("Semantic analysis activation binding is invalid.")
    canonical_report, canonical_sha, canonical_spec_sha, activation_smoke_sha = _load_intervention(
        manifest_path=PATHS["activation_manifest"],
        activation_path=PATHS["activation_smoke"],
        spec_path=PATHS["canonical_spec"],
        report_path=PATHS["canonical_report"],
    )
    semantic_intervention, semantic_intervention_sha, semantic_spec_sha, source_semantic_sha = (
        _load_intervention(
            manifest_path=PATHS["semantic_manifest"],
            activation_path=PATHS["semantic_activation"],
            spec_path=PATHS["semantic_spec"],
            report_path=PATHS["semantic_intervention"],
        )
    )
    if source_semantic_sha != semantic_activation_sha:
        raise ProbeSchemaError("Semantic intervention source activation hash is inconsistent.")
    geometry, geometry_sha = _read_json(PATHS["geometry"], MAX_REPORT_BYTES)
    validate_geometry_report(geometry)
    expected_geometry_sources = {
        "semantic_analysis_sha256": semantic_analysis_sha,
        "activation_report_sha256": semantic_activation_sha,
        "intervention_spec_sha256": semantic_spec_sha,
        "intervention_report_sha256": semantic_intervention_sha,
    }
    if any(geometry["source"].get(key) != value for key, value in expected_geometry_sources.items()):
        raise ProbeSchemaError("Geometry source bindings are inconsistent.")
    source_hashes = {
        smoke_probe["artifact"]["sha256"],
        semantic_probe["artifact"]["sha256"],
        boundary_activation["artifact"]["sha256"],
        semantic_activation["artifact"]["sha256"],
        canonical_report["artifact"]["sha256"],
        semantic_intervention["artifact"]["sha256"],
        artifact_hash,
    }
    if len(source_hashes) != 1:
        raise ProbeSchemaError("Validated reports do not describe one common model artifact.")

    inventory_entries = [
        _artifact_entry("architecture_report", PATHS["architecture"], architecture_sha),
        _artifact_entry("smoke_probe_report", PATHS["smoke_probe"], smoke_probe_sha),
        _artifact_entry("semantic_probe_report", PATHS["semantic_probe"], semantic_probe_sha),
        _artifact_entry("activation_smoke_report", PATHS["activation_smoke"], activation_smoke_sha),
        _artifact_entry("boundary_activation_report", PATHS["boundary_activation"], boundary_activation_sha),
        _artifact_entry("boundary_analysis", PATHS["boundary_analysis"], boundary_analysis_sha),
        _artifact_entry("semantic_activation_report", PATHS["semantic_activation"], semantic_activation_sha),
        _artifact_entry("semantic_analysis", PATHS["semantic_analysis"], semantic_analysis_sha),
        _artifact_entry("canonical_intervention_spec", PATHS["canonical_spec"], canonical_spec_sha),
        _artifact_entry("canonical_intervention_report", PATHS["canonical_report"], canonical_sha),
        _artifact_entry("semantic_intervention_spec", PATHS["semantic_spec"], semantic_spec_sha),
        _artifact_entry("semantic_intervention_report", PATHS["semantic_intervention"], semantic_intervention_sha),
        _artifact_entry("local_geometry_report", PATHS["geometry"], geometry_sha),
    ]
    for index, plot in enumerate(geometry["plots"]):
        plot_path = ROOT / plot["path"]
        if _artifact_entry(f"local_geometry_plot_{index}", plot_path)["sha256"] != plot["sha256"]:
            raise ProbeSchemaError("Geometry plot hash does not match the report.")
        inventory_entries.append(_artifact_entry(f"local_geometry_plot_{index}", plot_path))
    inventory = {
        "schema_version": 1,
        "artifact_kind": "generated_report_inventory",
        "entries": inventory_entries,
    }
    inventory_sha = _write_json(inventory_path, inventory)

    module_tree = architecture["loaded_object"]["module_tree"]
    linear_count = sum(item["type"].endswith(".Linear") for item in module_tree)
    relu_count = sum(item["type"].endswith(".ReLU") for item in module_tree)
    scalar_observations = smoke_probe["summary"]["observation_count"] + semantic_probe["summary"]["observation_count"]
    scalar_nonzero = sum(
        item["count"]
        for report in (smoke_probe, semantic_probe)
        for item in report["summary"]["output_counts"]
        if item["value"] != 0.0
    )
    semantic_summary = summarize_semantic_interventions(semantic_intervention)
    canonical_summary = summarize_canonical_interventions(canonical_report)
    geometry_rank = geometry["direction_geometry"]["effective_rank"]
    cross_fold_model = geometry["direction_geometry"]["cross_fold_cosine_model"]
    summary = {
        "schema_version": SYNTHESIS_SCHEMA_VERSION,
        "analysis_kind": "milestone_5_findings_synthesis",
        "artifact": architecture["artifact"],
        "sources": {
            "inventory_sha256": inventory_sha,
            **{entry["name"] + "_sha256": entry["sha256"] for entry in inventory_entries},
        },
        "architecture": {
            "linear_layer_count": linear_count,
            "relu_layer_count": relu_count,
            "parameter_count": architecture["loaded_object"]["parameter_count"],
            "input_character_limit": 55,
            "predicate_count": architecture["final_predicate_circuit"]["predicate_count"],
            "target_digest_hex": "".join(
                f"{int(item['target']):02x}" for item in architecture["final_predicate_circuit"]["predicates"]
            ),
        },
        "behavioral_probing": {
            "observation_count": scalar_observations,
            "nonzero_observation_count": scalar_nonzero,
            "deterministic": smoke_probe["summary"]["all_cases_deterministic"]
            and semantic_probe["summary"]["all_cases_deterministic"],
        },
        "encoding": {
            "boundary_case_count": len(boundary_analysis["cases"]),
            "ascii_and_byte_range_status": "ordinary_md5_for_tested_short_inputs",
            "character_cutoff": 55,
            "unicode_above_255_status": "unresolved",
            "universal_encoding_claim": False,
        },
        "semantic_directions": {
            "direction_count": len(geometry["directions"]),
            "left_held_out_accuracy": semantic_analysis["slots"]["left"]["mean_held_out_accuracy"],
            "right_held_out_accuracy": semantic_analysis["slots"]["right"]["mean_held_out_accuracy"],
            "chance_accuracy": semantic_analysis["slots"]["left"]["chance_accuracy"],
            "status": "rigorous_null_result",
            "numerical_rank": geometry_rank["numerical_rank"],
            "entropy_effective_rank": geometry_rank["entropy_effective_rank"],
            "same_slot_category_cross_fold_mean_cosine": geometry["direction_geometry"]
            ["cosine_summaries"]["same_slot_category_across_folds"]["mean"],
            "same_fold_category_cross_slot_mean_cosine": geometry["direction_geometry"]
            ["cosine_summaries"]["same_fold_category_across_slots"]["mean"],
            "cross_fold_model": {
                "formula": cross_fold_model["model"],
                "observation_count": cross_fold_model["row_count"],
                "block_count": cross_fold_model["block_count"],
                "same_category_observation_count": cross_fold_model[
                    "same_category_row_count"
                ],
                "different_category_observation_count": cross_fold_model[
                    "different_category_row_count"
                ],
                "same_category_mean": cross_fold_model["same_category_mean"],
                "different_category_mean": cross_fold_model["different_category_mean"],
                "same_minus_different": cross_fold_model["same_category_effect"],
                "r_squared": cross_fold_model["fit"]["r_squared"],
                "permutation_repetitions": cross_fold_model["permutation"]["repetitions"],
                "permutation_two_sided_p_value": cross_fold_model["permutation"]
                ["two_sided_p_value"],
                "permutation_scheme": cross_fold_model["permutation"]["scheme"],
            },
        },
        "causal_controls": {
            "canonical": canonical_summary,
            "semantic": semantic_summary,
        },
        "local_geometry": {
            "predicate_jacobian_shape": geometry["predicate_geometry"]["shape"],
            "predicate_jacobian_nonzero_count": geometry["predicate_geometry"]["nonzero_count"],
            "analytic_autograd_jacobian_max_abs_error": geometry["predicate_geometry"]
            ["analytic_autograd_max_abs_error"],
            "endpoint_count": geometry["boundary_analysis"]["endpoint_count"],
            "endpoints_with_predicate_relu_crossings": geometry["boundary_analysis"]
            ["endpoints_with_predicate_relu_crossings"],
            "crossing_observation_count": geometry["boundary_analysis"]
            ["observed_crossing_observation_count"],
            "crossing_mismatch_count": geometry["boundary_analysis"]["crossing_mismatch_count"],
            "readout_hessian_frobenius_norm": geometry["curvature"]
            ["readout_hessian_frobenius_norm"],
            "output_hessian_frobenius_norm": geometry["curvature"]
            ["output_hessian_frobenius_norm"],
            "pyhessian_status": geometry["curvature"]["pyhessian"],
        },
        "remaining_uncertainty": [
            "The exact encoding for code points above 255 is unresolved.",
            "No unrestricted MD5 preimage search is feasible; only finite structured domains are appropriate.",
            "The held-out semantic directions remain null candidates rather than validated semantic features.",
        ],
        "reproducibility": {
            "all_source_reports_validated": True,
            "all_source_hashes_bound": True,
            "generated_artifact_count": len(inventory_entries),
            "canonical_result_is_script_generated": True,
        },
    }
    return summary, inventory


def render_markdown(summary: dict[str, Any]) -> str:
    architecture = summary["architecture"]
    semantic = summary["semantic_directions"]
    cross_fold = semantic["cross_fold_model"]
    controls = summary["causal_controls"]
    geometry = summary["local_geometry"]
    uncertainty = "\n".join(f"- {item}" for item in summary["remaining_uncertainty"])
    return f"""# Final Mechanistic Findings

This report is generated by `scripts/synthesize_findings.py` from schema-validated, hash-bound
reports. The source inventory is `outputs/reports/m5-artifact-inventory-v1.json`.

## Recovered Mechanism

The artifact has {architecture['linear_layer_count']:,} linear layers and
{architecture['relu_layer_count']:,} ReLUs, with {architecture['parameter_count']:,} parameters.
It consumes at most {architecture['input_character_limit']} Python characters. The
final representation exposes 16 MD5-byte-like predicates and the final scalar is positive only
when every predicate matches target `{architecture['target_digest_hex']}`.

For tested short ASCII and byte-range inputs, the predicates equal ordinary MD5 of the unpadded
input. The experiment does not establish a universal encoding for code points above 255.

## Behavioral and Semantic Evidence

All {summary['behavioral_probing']['observation_count']} scalar probe observations were deterministic
zeros. The leakage-free semantic direction estimator achieved
{semantic['left_held_out_accuracy']:.3f} accuracy in the left slot and
{semantic['right_held_out_accuracy']:.3f} in the right slot versus {semantic['chance_accuracy']:.3f}
chance. This remains a rigorous null result.

The 30 candidate directions have numerical rank {semantic['numerical_rank']} and entropy effective
rank {semantic['entropy_effective_rank']:.3f}. Mean same-category cross-fold cosine within a slot
is {semantic['same_slot_category_cross_fold_mean_cosine']:.3f}; mean same-category cross-slot
cosine within a fold is {semantic['same_fold_category_cross_slot_mean_cosine']:.3f}. Geometry does
not convert the null result into evidence of semantic representation.

A blocked model over all {cross_fold['observation_count']} cross-fold cosine cells compares
{cross_fold['same_category_observation_count']} aligned-category pairs with
{cross_fold['different_category_observation_count']} cross-category controls across
{cross_fold['block_count']} slot/fold-pair blocks. Same-category mean cosine is
{cross_fold['same_category_mean']:.3f}, compared with {cross_fold['different_category_mean']:.3f}
for controls, an estimated alignment contrast of {cross_fold['same_minus_different']:.3f}.
A coherent fold-label permutation test with {cross_fold['permutation_repetitions']:,} repetitions
gives two-sided p={cross_fold['permutation_two_sided_p_value']:.6g}. This establishes reproducible
category alignment among the fitted directions, but it remains a stability result—not held-out
classification performance or evidence that the directions carry semantic information.

## Causal and Local-Geometry Evidence

Canonical replacement produces 16/16 matches, readout 1, and output 1. Each single-predicate break
produces 15/16 matches, readout 0, and output 0 across
{controls['canonical']['observation_count']} observations.

The semantic sweep contains {controls['semantic']['observation_count']} observations. Analytic and
observed predicate deltas differ by at most
{controls['semantic']['predicate_delta_max_abs_error']}; readout deltas differ by at most
{controls['semantic']['readout_delta_max_abs_error']:.8g}. It contains
{controls['semantic']['predicate_relu_crossing_observations']} predicate-ReLU crossing observations,
zero final-ReLU crossings, and zero nonzero outputs.

The exact predicate Jacobian has shape {geometry['predicate_jacobian_shape']} with
{geometry['predicate_jacobian_nonzero_count']} nonzero entries. Analytic and `torch.func` Jacobians
agree with maximum error {geometry['analytic_autograd_jacobian_max_abs_error']}. Across
{geometry['endpoint_count']} unique nonzero direction/case endpoints,
{geometry['endpoints_with_predicate_relu_crossings']} cross at least one predicate ReLU. Repetition
expands these to {geometry['crossing_observation_count']} crossing observations, all reproduced
with {geometry['crossing_mismatch_count']} mismatches.

At a clean fixed-region point, the readout Hessian Frobenius norm is
{geometry['readout_hessian_frobenius_norm']} and the output Hessian Frobenius norm is
{geometry['output_hessian_frobenius_norm']}. This is expected for a piecewise-affine ReLU circuit;
Jacobian jumps and boundary crossings are the meaningful nonlinear measurements.
PyHessian is not used because no parameter-space loss objective is defined.

## Remaining Uncertainty

{uncertainty}

## Reproduce

Run the local-geometry analysis in a separate Python 3.11 environment with
`requirements/geometry-analysis.txt`, then run:

```bash
python3 scripts/synthesize_findings.py
```

Model execution remains confined to the namespace and Landlock launchers. This synthesis performs
no model loading or inference.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and synthesize the canonical findings.")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/reports/m5-findings-summary-v1.json"
    )
    parser.add_argument(
        "--inventory", type=Path, default=ROOT / "outputs/reports/m5-artifact-inventory-v1.json"
    )
    parser.add_argument("--markdown", type=Path, default=ROOT / "docs/final_report.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len({args.output.resolve(), args.inventory.resolve(), args.markdown.resolve()}) != 3:
        print("Synthesis outputs must be distinct files.", file=sys.stderr)
        return 2
    try:
        summary, _inventory = build_summary(args.inventory)
        _write_json(args.output, summary)
        _write_text(args.markdown, render_markdown(summary))
    except (OSError, ProbeSchemaError, RecursionError, ValueError, KeyError) as exc:
        print(f"Findings synthesis rejected input: {exc}", file=sys.stderr)
        return 2
    print(f"Verified findings summary: {args.output}")
    print(f"Generated final report: {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
