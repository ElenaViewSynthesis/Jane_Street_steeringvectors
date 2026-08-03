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
    import matplotlib
    import numpy as np
    import scipy
    import scipy.linalg
    import seaborn as sns
    import torch
    from torch.func import hessian, jacrev
except ImportError as exc:  # pragma: no cover - exercised only in incomplete environments.
    print(
        "Local-geometry dependencies are unavailable; install requirements/geometry-analysis.txt "
        f"in a separate offline-analysis environment ({exc}).",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from scripts.activation_schema import PREDICATE_TARGETS
    from scripts.analyze_activation_report import load_validated_activation_report
    from scripts.intervention_schema import load_intervention_spec, validate_intervention_report
    from scripts.local_geometry_schema import (
        BOUNDARY_TOLERANCE,
        GEOMETRY_SCHEMA_VERSION,
        analytic_predicate_jacobian,
        cross_fold_cosine_rows,
        direction_records,
        load_semantic_analysis,
        predicate_boundaries,
        readout_gradient,
        validate_geometry_report,
    )
    from scripts.probe_schema import (
        ProbeSchemaError,
        load_probe_manifest,
        loads_json,
        read_bounded_regular_file,
        sha256_bytes,
    )
except ModuleNotFoundError:  # Direct execution from scripts/.
    from activation_schema import PREDICATE_TARGETS  # type: ignore[no-redef]
    from analyze_activation_report import load_validated_activation_report  # type: ignore[no-redef]
    from intervention_schema import (  # type: ignore[no-redef]
        load_intervention_spec,
        validate_intervention_report,
    )
    from local_geometry_schema import (  # type: ignore[no-redef]
        BOUNDARY_TOLERANCE,
        GEOMETRY_SCHEMA_VERSION,
        analytic_predicate_jacobian,
        cross_fold_cosine_rows,
        direction_records,
        load_semantic_analysis,
        predicate_boundaries,
        readout_gradient,
        validate_geometry_report,
    )
    from probe_schema import (  # type: ignore[no-redef]
        ProbeSchemaError,
        load_probe_manifest,
        loads_json,
        read_bounded_regular_file,
        sha256_bytes,
    )


MAX_SPEC_BYTES = 4 * 1024 * 1024
MAX_INTERVENTION_REPORT_BYTES = 64 * 1024 * 1024
DTYPE = torch.float64


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _tensor_values(tensor: torch.Tensor) -> Any:
    return tensor.detach().cpu().tolist()


def _matrix_stats(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
        "mean_absolute": sum(abs(value) for value in values) / len(values) if values else None,
    }


def _effective_rank(singular_values: torch.Tensor) -> dict[str, Any]:
    values = singular_values.detach().cpu()
    squared = values.square()
    total = float(squared.sum())
    probabilities = squared / total if total else squared
    nonzero = probabilities[probabilities > 0]
    entropy_rank = float(torch.exp(-(nonzero * torch.log(nonzero)).sum())) if len(nonzero) else 0.0
    participation = float(total * total / squared.square().sum()) if total else 0.0
    tolerance = max(30, 192) * torch.finfo(DTYPE).eps * float(values[0])
    cumulative = torch.cumsum(squared, dim=0) / total if total else squared

    def energy_rank(level: float) -> int:
        indices = torch.nonzero(cumulative >= level)
        return int(indices[0]) + 1 if len(indices) else 0

    return {
        "numerical_rank": int((values > tolerance).sum()),
        "numerical_rank_tolerance": tolerance,
        "entropy_effective_rank": entropy_rank,
        "participation_ratio": participation,
        "rank_for_90_percent_energy": energy_rank(0.90),
        "rank_for_95_percent_energy": energy_rank(0.95),
        "rank_for_99_percent_energy": energy_rank(0.99),
        "explained_energy": _tensor_values(cumulative),
    }


def _cosine_summaries(records: list[dict[str, Any]], gram: torch.Tensor) -> dict[str, Any]:
    cross_fold = []
    cross_slot = []
    off_diagonal = []
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            value = float(gram[left, right])
            off_diagonal.append(value)
            a, b = records[left], records[right]
            if a["slot"] == b["slot"] and a["category"] == b["category"]:
                cross_fold.append(value)
            if (
                a["slot"] != b["slot"]
                and a["fold_index"] == b["fold_index"]
                and a["category"] == b["category"]
            ):
                cross_slot.append(value)
    return {
        "same_slot_category_across_folds": _matrix_stats(cross_fold),
        "same_fold_category_across_slots": _matrix_stats(cross_slot),
        "all_off_diagonal": _matrix_stats(off_diagonal),
    }


def _cross_fold_cosine_model(
    records: list[dict[str, Any]],
    gram: torch.Tensor,
    *,
    permutation_repetitions: int = 10_000,
    permutation_seed: int = 20_260_803,
) -> dict[str, Any]:
    rows = cross_fold_cosine_rows(records, _tensor_values(gram))
    block_ids = sorted({row["block_id"] for row in rows})
    categories = sorted({row["category_a"] for row in rows})
    block_index = {block_id: index for index, block_id in enumerate(block_ids)}
    category_index = {category: index for index, category in enumerate(categories)}
    response = np.asarray([row["cosine"] for row in rows], dtype=np.float64)
    design = np.zeros((len(rows), 1 + (len(block_ids) - 1) + 1), dtype=np.float64)
    design[:, 0] = 1.0
    for row_index, row in enumerate(rows):
        current_block = block_index[row["block_id"]]
        if current_block:
            design[row_index, current_block] = 1.0
        design[row_index, -1] = float(row["same_category"])
    coefficients, _residuals, design_rank, singular_values = scipy.linalg.lstsq(design, response)
    fitted = design @ coefficients
    residual = response - fitted
    total_sum_squares = float(np.square(response - response.mean()).sum())
    residual_sum_squares = float(np.square(residual).sum())
    r_squared = 1.0 - residual_sum_squares / total_sum_squares
    same_values = response[design[:, -1] == 1.0]
    different_values = response[design[:, -1] == 0.0]
    observed_effect = float(coefficients[-1])
    direct_effect = float(same_values.mean() - different_values.mean())
    if not math.isclose(observed_effect, direct_effect, rel_tol=0.0, abs_tol=1e-12):
        raise ProbeSchemaError("Blocked cross-fold coefficient is inconsistent with balanced means.")

    blocks = []
    block_matrices = []
    block_matrix_lookup = {}
    for block_id in block_ids:
        block_rows = [row for row in rows if row["block_id"] == block_id]
        matrix = np.zeros((len(categories), len(categories)), dtype=np.float64)
        for row in block_rows:
            matrix[category_index[row["category_a"]], category_index[row["category_b"]]] = row[
                "cosine"
            ]
        diagonal = np.diag(matrix)
        off_diagonal = matrix[~np.eye(len(categories), dtype=bool)]
        block_number = block_index[block_id]
        different_intercept = float(coefficients[0]) + (
            float(coefficients[block_number]) if block_number else 0.0
        )
        blocks.append(
            {
                "block_id": block_id,
                "slot": block_rows[0]["slot"],
                "fold_a": block_rows[0]["fold_a"],
                "fold_b": block_rows[0]["fold_b"],
                "matrix": matrix.tolist(),
                "same_category_mean": float(diagonal.mean()),
                "different_category_mean": float(off_diagonal.mean()),
                "same_minus_different": float(diagonal.mean() - off_diagonal.mean()),
                "fitted_different_category_mean": different_intercept,
                "fitted_same_category_mean": different_intercept + observed_effect,
            }
        )
        block_matrices.append(matrix)
        block_matrix_lookup[(block_rows[0]["slot"], block_rows[0]["fold_a"], block_rows[0]["fold_b"])] = matrix

    generator = np.random.default_rng(permutation_seed)
    null_effects = np.empty(permutation_repetitions, dtype=np.float64)
    total_sum = float(sum(matrix.sum() for matrix in block_matrices))
    same_count = len(block_matrices) * len(categories)
    different_count = len(block_matrices) * (len(categories) ** 2 - len(categories))
    row_indices = np.arange(len(categories))
    for repetition in range(permutation_repetitions):
        selected_sum = 0.0
        for slot in ("left", "right"):
            fold_1_mapping = generator.permutation(len(categories))
            fold_2_mapping = generator.permutation(len(categories))
            selected_sum += float(
                block_matrix_lookup[(slot, 0, 1)][row_indices, fold_1_mapping].sum()
            )
            selected_sum += float(
                block_matrix_lookup[(slot, 0, 2)][row_indices, fold_2_mapping].sum()
            )
            selected_sum += float(
                block_matrix_lookup[(slot, 1, 2)][fold_1_mapping, fold_2_mapping].sum()
            )
        null_effects[repetition] = (
            selected_sum / same_count - (total_sum - selected_sum) / different_count
        )
    extreme_count = int(np.count_nonzero(np.abs(null_effects) >= abs(observed_effect)))
    permutation_p = (extreme_count + 1.0) / (permutation_repetitions + 1.0)
    null_standard_deviation = float(null_effects.std(ddof=1))

    stability = []
    for slot in ("left", "right"):
        for category in categories:
            values = [
                row["cosine"]
                for row in rows
                if row["slot"] == slot
                and row["same_category"]
                and row["category_a"] == category
            ]
            stability.append(
                {
                    "slot": slot,
                    "category": category,
                    "pairwise_cosines": values,
                    "mean": sum(values) / len(values),
                    "minimum": min(values),
                    "maximum": max(values),
                }
            )

    return {
        "model": "cosine ~ slot_by_fold_pair_fixed_effects + same_category",
        "interpretation": (
            "A blocked stability contrast. The permutation result tests category alignment across "
            "folds; it does not test held-out classification accuracy or establish semantics."
        ),
        "row_count": len(rows),
        "block_count": len(block_ids),
        "same_category_row_count": int(design[:, -1].sum()),
        "different_category_row_count": int(len(rows) - design[:, -1].sum()),
        "same_category_mean": float(same_values.mean()),
        "different_category_mean": float(different_values.mean()),
        "same_category_effect": observed_effect,
        "fixed_effect_coefficients": {
            "reference_block": block_ids[0],
            "reference_different_category_intercept": float(coefficients[0]),
            "block_offsets": {
                block_id: (0.0 if index == 0 else float(coefficients[index]))
                for block_id, index in block_index.items()
            },
            "same_category": observed_effect,
        },
        "fit": {
            "design_rank": int(design_rank),
            "design_singular_values": singular_values.tolist(),
            "residual_sum_squares": residual_sum_squares,
            "r_squared": r_squared,
        },
        "permutation": {
            "scheme": (
                "permute category labels coherently across folds within each slot, with fold 0 "
                "anchored and shared fold mappings preserved across pairwise matrices"
            ),
            "seed": permutation_seed,
            "repetitions": permutation_repetitions,
            "two_sided_p_value": permutation_p,
            "extreme_count": extreme_count,
            "null_mean": float(null_effects.mean()),
            "null_standard_deviation": null_standard_deviation,
            "observed_effect_in_null_standard_deviations": (
                observed_effect / null_standard_deviation if null_standard_deviation else None
            ),
            "null_quantiles": {
                "q025": float(np.quantile(null_effects, 0.025)),
                "q500": float(np.quantile(null_effects, 0.5)),
                "q975": float(np.quantile(null_effects, 0.975)),
            },
            "null_float64_sha256": hashlib.sha256(null_effects.astype("<f8").tobytes()).hexdigest(),
        },
        "blocks": blocks,
        "category_stability": stability,
        "rows": rows,
    }


def _tail_functions(jacobian: torch.Tensor) -> tuple[Any, Any, Any, Any]:
    targets = torch.tensor(PREDICATE_TARGETS, dtype=jacobian.dtype)

    def predicates(h192: torch.Tensor) -> torch.Tensor:
        return jacobian @ h192

    def stages(h192: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        values = predicates(h192)
        a48 = torch.cat((values - (targets + 1.0), values - targets, values - (targets - 1.0)))
        z48 = torch.relu(a48)
        indicators = z48[:16] - 2.0 * z48[16:32] + z48[32:]
        readout = indicators.sum() - 15.0
        return values, a48, readout, torch.relu(readout)

    def readout(h192: torch.Tensor) -> torch.Tensor:
        return stages(h192)[2]

    def output(h192: torch.Tensor) -> torch.Tensor:
        return stages(h192)[3]

    return predicates, stages, readout, output


def _load_intervention_evidence(
    *,
    manifest_path: Path,
    activation_report: dict[str, Any],
    activation_report_sha256: str,
    spec_path: Path,
    report_path: Path,
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    manifest, _payload, manifest_sha256 = load_probe_manifest(manifest_path)
    spec_payload = read_bounded_regular_file(spec_path, MAX_SPEC_BYTES)
    spec, spec_sha256 = load_intervention_spec(
        spec_payload, case_ids={case["id"] for case in manifest["cases"]}
    )
    if spec["source_manifest_sha256"] != manifest_sha256:
        raise ProbeSchemaError("Intervention spec is not bound to the supplied manifest.")
    if spec["source_activation_report_sha256"] != activation_report_sha256:
        raise ProbeSchemaError("Intervention spec is not bound to the activation report.")
    report_payload = read_bounded_regular_file(report_path, MAX_INTERVENTION_REPORT_BYTES)
    report_sha256 = sha256_bytes(report_payload)
    report = loads_json(report_payload)
    repetitions = report.get("probe_suite", {}).get("repetitions") if isinstance(report, dict) else None
    validated = validate_intervention_report(
        report,
        manifest=manifest,
        expected_model_sha256=activation_report["artifact"]["sha256"],
        expected_spec_sha256=spec_sha256,
        spec=spec,
        expected_source_report_sha256=activation_report_sha256,
        repetitions=repetitions,
    )
    return spec, spec_sha256, validated, report_sha256


def _observed_crossings(report: dict[str, Any]) -> tuple[dict[tuple[str, str, float], list[list[int]]], int]:
    lookup: dict[tuple[str, str, float], list[list[int]]] = {}
    crossing_observations = 0
    for result in report["results"]:
        case_id = result["case"]["id"]
        for group in result["interventions"]:
            rows = [
                item["comparison"]["predicate_relu_crossing_rows"]
                for item in group["observations"]
            ]
            lookup[(case_id, group["intervention_id"], float(group["strength"]))] = rows
            crossing_observations += sum(bool(item) for item in rows)
    return lookup, crossing_observations


def _plot_matrix(matrix: np.ndarray, labels: list[str], title: str, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 10))
    sns.heatmap(matrix, cmap="vlag", center=0.0, vmin=-1.0, vmax=1.0, ax=axis,
                xticklabels=labels, yticklabels=labels, square=True, cbar_kws={"shrink": 0.75})
    axis.set_title(title)
    axis.tick_params(axis="x", labelrotation=90, labelsize=6)
    axis.tick_params(axis="y", labelrotation=0, labelsize=6)
    figure.tight_layout()
    figure.savefig(output, dpi=160, metadata={"Software": "jsmi-analyze-local-geometry-v1"})
    plt.close(figure)


def _write_plots(
    plot_directory: Path,
    records: list[dict[str, Any]],
    direction_gram: torch.Tensor,
    response_gram: torch.Tensor,
    singular_values: torch.Tensor,
    cross_fold_model: dict[str, Any],
) -> list[dict[str, Any]]:
    plot_directory.mkdir(parents=True, exist_ok=True)
    labels = [
        f"{item['slot'][0]}{item['fold_index']}-{item['category']}" for item in records
    ]
    paths = [
        plot_directory / "m5-direction-gram-v1.png",
        plot_directory / "m5-response-gram-v1.png",
        plot_directory / "m5-direction-singular-values-v1.png",
        plot_directory / "m5-cross-fold-cosine-model-v1.png",
    ]
    _plot_matrix(direction_gram.detach().cpu().numpy(), labels, "Direction Gram matrix", paths[0])
    response = response_gram.detach().cpu().numpy()
    response_scale = np.sqrt(np.diag(response))
    denominator = np.outer(response_scale, response_scale)
    response_cosine = np.divide(response, denominator, out=np.zeros_like(response), where=denominator > 0)
    _plot_matrix(response_cosine, labels, "Predicate-response cosine matrix", paths[1])
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(range(1, len(singular_values) + 1), singular_values.detach().cpu().numpy(), marker="o")
    axis.set(xlabel="Singular-value index", ylabel="Singular value", title="Semantic direction spectrum")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(paths[2], dpi=160, metadata={"Software": "jsmi-analyze-local-geometry-v1"})
    plt.close(figure)
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    categories = sorted({row["category_a"] for row in cross_fold_model["rows"]})
    for axis, block in zip(axes.flat, cross_fold_model["blocks"]):
        sns.heatmap(
            np.asarray(block["matrix"]),
            cmap="vlag",
            center=0.0,
            vmin=-0.5,
            vmax=0.6,
            annot=True,
            fmt=".2f",
            xticklabels=categories,
            yticklabels=categories,
            cbar=False,
            ax=axis,
        )
        axis.set_title(block["block_id"], fontsize=9)
        axis.tick_params(axis="x", labelrotation=45, labelsize=7)
        axis.tick_params(axis="y", labelrotation=0, labelsize=7)
    figure.suptitle("Cross-fold direction cosines by category mapping")
    figure.tight_layout()
    figure.savefig(paths[3], dpi=160, metadata={"Software": "jsmi-analyze-local-geometry-v1"})
    plt.close(figure)
    return [
        {"path": str(path), "sha256": _sha256_file(path), "size_bytes": path.stat().st_size}
        for path in paths
    ]


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    semantic, semantic_sha256 = load_semantic_analysis(args.semantic_analysis)
    activation, activation_sha256 = load_validated_activation_report(
        args.activation_report, args.manifest
    )
    if semantic["source"]["activation_report_sha256"] != activation_sha256:
        raise ProbeSchemaError("Semantic analysis is not bound to the activation report.")
    if semantic["source"]["artifact_sha256"] != activation["artifact"]["sha256"]:
        raise ProbeSchemaError("Semantic analysis artifact hash does not match the activation report.")
    spec, spec_sha256, intervention_report, intervention_report_sha256 = _load_intervention_evidence(
        manifest_path=args.manifest,
        activation_report=activation,
        activation_report_sha256=activation_sha256,
        spec_path=args.intervention_spec,
        report_path=args.intervention_report,
    )
    records = direction_records(semantic)
    spec_by_id = {item["id"]: item for item in spec["interventions"]}
    if set(spec_by_id) != {item["id"] for item in records}:
        raise ProbeSchemaError("Semantic direction analysis and intervention spec ids differ.")

    direction_matrix = torch.tensor([item["vector"] for item in records], dtype=DTYPE)
    jacobian = torch.tensor(analytic_predicate_jacobian(), dtype=DTYPE)
    predicates, stages, readout_fn, output_fn = _tail_functions(jacobian)
    autograd_jacobian = jacrev(predicates)(torch.zeros(192, dtype=DTYPE))
    jacobian_error = float((autograd_jacobian - jacobian).abs().max())
    if jacobian_error != 0.0:
        raise ProbeSchemaError("Analytic and autograd predicate Jacobians disagree.")

    direction_gram = direction_matrix @ direction_matrix.T
    responses = direction_matrix @ jacobian.T
    response_gram = responses @ responses.T
    singular_values = torch.linalg.svdvals(direction_matrix)
    scipy_singular_values = scipy.linalg.svdvals(direction_matrix.detach().cpu().numpy())
    scipy_error = float(np.max(np.abs(scipy_singular_values - singular_values.detach().cpu().numpy())))
    cross_fold_model = _cross_fold_cosine_model(records, direction_gram)

    activation_by_case = {
        result["case"]["id"]: result["observations"][0] for result in activation["results"]
    }
    selected_case_ids = sorted({case_id for item in records for case_id in item["case_ids"]})
    baseline_boundaries = []
    for case_id in selected_case_ids:
        h_values = activation_by_case[case_id]["tensors"]["h192"]["values"]
        boundaries = predicate_boundaries(h_values)
        nearest = min(item["euclidean_distance"] for item in boundaries)
        baseline_boundaries.append(
            {
                "case_id": case_id,
                "nearest_euclidean_distance": nearest,
                "nearest_rows": [
                    item["row"]
                    for item in boundaries
                    if math.isclose(item["euclidean_distance"], nearest, rel_tol=0.0, abs_tol=1e-15)
                ],
                "on_boundary_rows": [item["row"] for item in boundaries if item["on_boundary"]],
            }
        )

    observed_lookup, observed_crossing_observations = _observed_crossings(intervention_report)
    endpoint_records = []
    mismatch_count = 0
    analytic_crossing_observations = 0
    quantization_errors = []
    for direction_index, record in enumerate(records):
        spec_item = spec_by_id[record["id"]]
        execution_direction = torch.tensor(spec_item["vector"], dtype=torch.float32)
        quantization_errors.append(
            max(abs(left - right) for left, right in zip(record["vector"], spec_item["vector"]))
        )
        for case_id in record["case_ids"]:
            observation = activation_by_case[case_id]
            h0 = torch.tensor(observation["tensors"]["h192"]["values"], dtype=torch.float32)
            a0_observed = torch.tensor(observation["tensors"]["a48"]["values"], dtype=torch.float32)
            readout0 = float(observation["tensors"]["readout"]["values"][0])
            gradient0 = torch.tensor(readout_gradient(_tensor_values(h0.to(DTYPE))), dtype=DTYPE)
            output_gradient0 = gradient0 if readout0 > 0.0 else torch.zeros_like(gradient0)
            for strength in (-1.0, 1.0):
                h1 = h0 + float(strength) * execution_direction
                _p1, a1, readout1_tensor, _output1 = stages(h1.to(DTYPE))
                a1_float32 = a1.to(torch.float32)
                changed_rows = [
                    index
                    for index, (before, after) in enumerate(zip(a0_observed, a1_float32))
                    if bool(before > 0.0) != bool(after > 0.0)
                ]
                observed = observed_lookup[(case_id, record["id"], strength)]
                if any(rows != changed_rows for rows in observed):
                    mismatch_count += 1
                analytic_crossing_observations += len(observed) * int(bool(changed_rows))
                gradient1 = torch.tensor(readout_gradient(_tensor_values(h1.to(DTYPE))), dtype=DTYPE)
                readout1 = float(readout1_tensor)
                output_gradient1 = gradient1 if readout1 > 0.0 else torch.zeros_like(gradient1)
                readout_jump = gradient1 - gradient0
                output_jump = output_gradient1 - output_gradient0
                directional_crossings = []
                p0 = predicates(h0.to(DTYPE))
                direction_response = jacobian @ execution_direction.to(DTYPE)
                for row in changed_rows:
                    predicate_index = row % 16
                    group = row // 16
                    offset = (1.0, 0.0, -1.0)[group]
                    residual = float(p0[predicate_index] - (PREDICATE_TARGETS[predicate_index] + offset))
                    slope = float(direction_response[predicate_index])
                    alpha = -residual / slope if slope else None
                    directional_crossings.append(
                        {
                            "row": row,
                            "alpha": alpha,
                            "baseline_on_boundary": abs(residual) <= BOUNDARY_TOLERANCE,
                        }
                    )
                endpoint_records.append(
                    {
                        "direction_index": direction_index,
                        "direction_id": record["id"],
                        "case_id": case_id,
                        "strength": strength,
                        "changed_relu_rows": changed_rows,
                        "directional_crossings": directional_crossings,
                        "readout_jacobian_jump_norm": float(torch.linalg.vector_norm(readout_jump)),
                        "readout_jacobian_jump": _tensor_values(readout_jump) if changed_rows else None,
                        "output_jacobian_jump_norm": float(torch.linalg.vector_norm(output_jump)),
                        "final_relu_crossed": (readout0 > 0.0) != (readout1 > 0.0),
                    }
                )
    if mismatch_count:
        raise ProbeSchemaError(
            f"Analytic endpoint crossings disagree with {mismatch_count} intervention groups."
        )
    if analytic_crossing_observations != observed_crossing_observations:
        raise ProbeSchemaError("Analytic and observed crossing totals disagree.")

    clean_case_id = next(
        case_id
        for case_id in selected_case_ids
        if not any(
            abs(value) <= BOUNDARY_TOLERANCE
            for value in activation_by_case[case_id]["tensors"]["a48"]["values"]
        )
    )
    clean_h = torch.tensor(
        activation_by_case[clean_case_id]["tensors"]["h192"]["values"], dtype=DTYPE
    )
    h_readout = hessian(readout_fn)(clean_h)
    h_output = hessian(output_fn)(clean_h)
    representative_gradient = jacrev(readout_fn)(clean_h)
    analytic_representative = torch.tensor(
        readout_gradient(_tensor_values(clean_h)), dtype=DTYPE
    )
    gradient_error = float((representative_gradient - analytic_representative).abs().max())
    if gradient_error != 0.0:
        raise ProbeSchemaError("Analytic and autograd readout Jacobians disagree.")

    plots = _write_plots(
        args.plot_directory,
        records,
        direction_gram,
        response_gram,
        singular_values,
        cross_fold_model,
    )
    report = {
        "schema_version": GEOMETRY_SCHEMA_VERSION,
        "analysis_kind": "relu_local_geometry",
        "source": {
            "semantic_analysis_sha256": semantic_sha256,
            "activation_report_sha256": activation_sha256,
            "manifest_sha256": semantic["source"]["manifest_sha256"],
            "artifact_sha256": semantic["source"]["artifact_sha256"],
            "intervention_spec_sha256": spec_sha256,
            "intervention_report_sha256": intervention_report_sha256,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": sns.__version__,
            "dtype": "torch.float64",
        },
        "directions": records,
        "predicate_geometry": {
            "jacobian": _tensor_values(jacobian),
            "jacobian_canonical_sha256": _canonical_sha256(_tensor_values(jacobian)),
            "shape": [16, 192],
            "nonzero_count": int(torch.count_nonzero(jacobian)),
            "analytic_autograd_max_abs_error": jacobian_error,
        },
        "direction_geometry": {
            "direction_matrix_shape": [30, 192],
            "direction_gram": _tensor_values(direction_gram),
            "predicate_responses": _tensor_values(responses),
            "response_gram": _tensor_values(response_gram),
            "cosine_summaries": _cosine_summaries(records, direction_gram),
            "cross_fold_cosine_model": cross_fold_model,
            "singular_values": _tensor_values(singular_values),
            "effective_rank": _effective_rank(singular_values),
            "scipy_singular_value_max_abs_error": scipy_error,
            "float32_spec_quantization_max_abs_error": max(quantization_errors),
        },
        "boundary_analysis": {
            "baseline_case_count": len(baseline_boundaries),
            "baseline_boundaries": baseline_boundaries,
            "endpoint_count": len(endpoint_records),
            "endpoints_with_predicate_relu_crossings": sum(
                bool(item["changed_relu_rows"]) for item in endpoint_records
            ),
            "analytic_crossing_observation_count": analytic_crossing_observations,
            "observed_crossing_observation_count": observed_crossing_observations,
            "crossing_mismatch_count": mismatch_count,
            "final_relu_crossing_count": sum(item["final_relu_crossed"] for item in endpoint_records),
            "endpoints": endpoint_records,
        },
        "curvature": {
            "representative_case_id": clean_case_id,
            "representative_point_has_predicate_boundary": False,
            "readout_jacobian_analytic_autograd_max_abs_error": gradient_error,
            "readout_hessian_frobenius_norm": float(torch.linalg.matrix_norm(h_readout)),
            "readout_hessian_max_abs_entry": float(h_readout.abs().max()),
            "output_hessian_frobenius_norm": float(torch.linalg.matrix_norm(h_output)),
            "output_hessian_max_abs_entry": float(h_output.abs().max()),
            "loss_hessian": {
                "status": "not_computed",
                "reason": "No scientifically justified scalar loss is defined for this puzzle output.",
            },
            "pyhessian": {
                "status": "not_used",
                "reason": (
                    "PyHessian targets parameter-space loss eigenvalues, trace, and spectral "
                    "density; this report differentiates the recovered scalar tail with respect "
                    "to h192 and has no parameter-loss objective."
                ),
            },
        },
        "plots": plots,
        "conclusion": {
            "semantic_status": "candidate_directions_from_a_held_out_null_result",
            "primary_nonlinearity_measure": "jacobian_jumps_and_relu_boundary_crossings",
            "ordinary_output_hessian_interpretation": (
                "Zero inside the tested fixed ReLU region; it does not represent boundary kinks."
            ),
        },
    }
    return validate_geometry_report(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze exact local geometry of the recovered h192-to-output ReLU circuit."
    )
    parser.add_argument("--semantic-analysis", type=Path, required=True)
    parser.add_argument("--activation-report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--intervention-spec", type=Path, required=True)
    parser.add_argument("--intervention-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot-directory", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = {
        args.semantic_analysis.resolve(),
        args.activation_report.resolve(),
        args.manifest.resolve(),
        args.intervention_spec.resolve(),
        args.intervention_report.resolve(),
    }
    if args.output.resolve() in inputs:
        print("Local-geometry output cannot overwrite an input file.", file=sys.stderr)
        return 2
    try:
        report = analyze(args)
    except (OSError, ProbeSchemaError, RecursionError, RuntimeError, ValueError) as exc:
        print(f"Local-geometry analysis rejected input: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=args.output.parent,
        prefix=f".{args.output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(report, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary_path, args.output)
    print(f"Verified local-geometry report: {args.output}")
    print(
        "Directions: 30; endpoints: "
        f"{report['boundary_analysis']['endpoint_count']}; crossing observations: "
        f"{report['boundary_analysis']['observed_crossing_observation_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
