from __future__ import annotations

import math
from pathlib import Path
from typing import Any

try:
    from scripts.activation_schema import PREDICATE_TARGETS
    from scripts.probe_schema import (
        ProbeSchemaError,
        loads_json,
        read_bounded_regular_file,
        sha256_bytes,
    )
except ModuleNotFoundError:  # Direct execution from scripts/.
    from activation_schema import PREDICATE_TARGETS  # type: ignore[no-redef]
    from probe_schema import (  # type: ignore[no-redef]
        ProbeSchemaError,
        loads_json,
        read_bounded_regular_file,
        sha256_bytes,
    )


GEOMETRY_SCHEMA_VERSION = 1
SEMANTIC_ANALYSIS_SCHEMA_VERSION = 1
H192_WIDTH = 192
PREDICATE_COUNT = 16
MAX_SEMANTIC_ANALYSIS_BYTES = 4 * 1024 * 1024
VECTOR_NORM_TOLERANCE = 1e-9
BOUNDARY_TOLERANCE = 1e-10


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_vector(value: Any, width: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != width:
        raise ProbeSchemaError(f"{label} must contain exactly {width} values.")
    vector = []
    for item in value:
        if type(item) not in {int, float} or not math.isfinite(float(item)):
            raise ProbeSchemaError(f"{label} must contain finite numeric values.")
        vector.append(float(item))
    return vector


def validate_semantic_analysis(document: Any) -> dict[str, Any]:
    required = {
        "schema_version",
        "analysis_kind",
        "source",
        "protocol",
        "categories",
        "words_by_category",
        "slots",
        "interpretation",
    }
    if not isinstance(document, dict) or set(document) != required:
        raise ProbeSchemaError("Semantic analysis has an invalid top-level field set.")
    if document["schema_version"] != SEMANTIC_ANALYSIS_SCHEMA_VERSION:
        raise ProbeSchemaError("Semantic analysis schema version is unsupported.")
    if document["analysis_kind"] != "held_out_semantic_directions":
        raise ProbeSchemaError("Semantic analysis kind is unsupported.")
    source = document["source"]
    source_fields = {
        "activation_report_sha256",
        "artifact_sha256",
        "manifest_sha256",
        "suite_id",
    }
    if not isinstance(source, dict) or set(source) != source_fields:
        raise ProbeSchemaError("Semantic analysis source metadata is invalid.")
    if not all(_is_sha256(source[key]) for key in source_fields - {"suite_id"}):
        raise ProbeSchemaError("Semantic analysis source hashes are invalid.")
    if not isinstance(source["suite_id"], str) or not source["suite_id"]:
        raise ProbeSchemaError("Semantic analysis suite id is invalid.")
    categories = document["categories"]
    if (
        not isinstance(categories, list)
        or len(categories) != 5
        or categories != sorted(set(categories))
        or any(not isinstance(category, str) or not category for category in categories)
    ):
        raise ProbeSchemaError("Semantic categories must be five unique sorted labels.")
    if set(document["slots"]) != {"left", "right"}:
        raise ProbeSchemaError("Semantic analysis must contain left and right slots.")
    seen_ids: set[str] = set()
    for slot in ("left", "right"):
        slot_result = document["slots"][slot]
        if not isinstance(slot_result, dict) or not isinstance(slot_result.get("folds"), list):
            raise ProbeSchemaError("Semantic slot result is invalid.")
        folds = slot_result["folds"]
        if len(folds) != 3 or [fold.get("fold_index") for fold in folds] != [0, 1, 2]:
            raise ProbeSchemaError("Semantic analysis must contain the three declared folds.")
        for fold in folds:
            directions = fold.get("directions")
            if not isinstance(directions, dict) or set(directions) != set(categories):
                raise ProbeSchemaError("Semantic fold directions do not cover every category.")
            evaluation = fold.get("evaluation")
            predictions = evaluation.get("predictions") if isinstance(evaluation, dict) else None
            if not isinstance(predictions, list) or not predictions:
                raise ProbeSchemaError("Semantic fold evaluation predictions are missing.")
            prediction_ids: set[str] = set()
            for prediction in predictions:
                if not isinstance(prediction, dict):
                    raise ProbeSchemaError("Semantic prediction entry is invalid.")
                case_id = prediction.get("case_id")
                if not isinstance(case_id, str) or not case_id or case_id in prediction_ids:
                    raise ProbeSchemaError("Semantic prediction case ids must be unique per fold.")
                prediction_ids.add(case_id)
                if prediction.get("actual") not in categories:
                    raise ProbeSchemaError("Semantic prediction category is invalid.")
            for category in categories:
                identifier = f"add-{slot}-fold-{fold['fold_index']}-{category}"
                if identifier in seen_ids:
                    raise ProbeSchemaError("Semantic direction identifier is duplicated.")
                seen_ids.add(identifier)
                vector = _finite_vector(
                    directions[category], H192_WIDTH, f"Semantic direction {identifier!r}"
                )
                norm = math.sqrt(sum(value * value for value in vector))
                if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=VECTOR_NORM_TOLERANCE):
                    raise ProbeSchemaError("Semantic direction is not L2-normalized.")
                selected = [item for item in predictions if item["actual"] == category]
                if len(selected) != 10:
                    raise ProbeSchemaError(
                        "Every semantic direction must have ten held-out category cases."
                    )
    if len(seen_ids) != 30:
        raise ProbeSchemaError("Semantic analysis must define exactly thirty directions.")
    return document


def load_semantic_analysis(path: Path) -> tuple[dict[str, Any], str]:
    payload = read_bounded_regular_file(path, MAX_SEMANTIC_ANALYSIS_BYTES)
    return validate_semantic_analysis(loads_json(payload)), sha256_bytes(payload)


def direction_records(document: dict[str, Any]) -> list[dict[str, Any]]:
    validated = validate_semantic_analysis(document)
    records = []
    for slot in ("left", "right"):
        for fold in validated["slots"][slot]["folds"]:
            for category in validated["categories"]:
                case_ids = [
                    prediction["case_id"]
                    for prediction in fold["evaluation"]["predictions"]
                    if prediction["actual"] == category
                ]
                records.append(
                    {
                        "id": f"add-{slot}-fold-{fold['fold_index']}-{category}",
                        "slot": slot,
                        "fold_index": fold["fold_index"],
                        "category": category,
                        "vector": [float(value) for value in fold["directions"][category]],
                        "case_ids": case_ids,
                    }
                )
    return records


def analytic_predicate_jacobian() -> list[list[float]]:
    weights = [float(2**bit) for bit in range(8)]
    matrix = [[0.0] * H192_WIDTH for _ in range(PREDICATE_COUNT)]

    def set_block(row: int, block: int, scale: float) -> None:
        start = block * 8
        matrix[row][start : start + 8] = [scale * value for value in weights]

    for offset in range(4):
        set_block(offset, offset, 1.0)
        set_block(4 + offset, 4 + offset, 1.0)
        set_block(4 + offset, 8 + offset, -2.0)
        set_block(8 + offset, 12 + offset, 1.0)
        set_block(12 + offset, 16 + offset, 1.0)
        set_block(12 + offset, 20 + offset, -2.0)
    return matrix


def matrix_vector(matrix: list[list[float]], vector: list[float]) -> list[float]:
    if not matrix or any(len(row) != len(vector) for row in matrix):
        raise ProbeSchemaError("Matrix and vector dimensions do not align.")
    return [sum(left * right for left, right in zip(row, vector)) for row in matrix]


def gram_matrix(matrix: list[list[float]]) -> list[list[float]]:
    if not matrix or any(len(row) != len(matrix[0]) for row in matrix):
        raise ProbeSchemaError("Gram input must be a nonempty rectangular matrix.")
    return [
        [sum(a * b for a, b in zip(left, right)) for right in matrix]
        for left in matrix
    ]


def cross_fold_cosine_rows(
    records: list[dict[str, Any]], gram: list[list[float]]
) -> list[dict[str, Any]]:
    if len(records) != 30 or len(gram) != 30 or any(len(row) != 30 for row in gram):
        raise ProbeSchemaError("Cross-fold cosine modeling requires thirty directions and a 30 x 30 Gram matrix.")
    index: dict[tuple[str, int, str], int] = {}
    for position, record in enumerate(records):
        key = (record.get("slot"), record.get("fold_index"), record.get("category"))
        if (
            key[0] not in {"left", "right"}
            or key[1] not in {0, 1, 2}
            or not isinstance(key[2], str)
            or key in index
        ):
            raise ProbeSchemaError("Direction records do not form the required slot/fold/category grid.")
        index[key] = position
    categories = sorted({key[2] for key in index})
    if len(categories) != 5 or len(index) != 30:
        raise ProbeSchemaError("Direction records must contain five categories in every slot and fold.")
    rows = []
    for slot in ("left", "right"):
        for fold_a, fold_b in ((0, 1), (0, 2), (1, 2)):
            block_id = f"{slot}-fold-{fold_a}-vs-{fold_b}"
            for category_a in categories:
                for category_b in categories:
                    left = index[(slot, fold_a, category_a)]
                    right = index[(slot, fold_b, category_b)]
                    value = gram[left][right]
                    if type(value) not in {int, float} or not math.isfinite(float(value)):
                        raise ProbeSchemaError("Cross-fold Gram entries must be finite numeric values.")
                    rows.append(
                        {
                            "block_id": block_id,
                            "slot": slot,
                            "fold_a": fold_a,
                            "fold_b": fold_b,
                            "category_a": category_a,
                            "category_b": category_b,
                            "same_category": category_a == category_b,
                            "cosine": float(value),
                        }
                    )
    if len(rows) != 150 or sum(row["same_category"] for row in rows) != 30:
        raise ProbeSchemaError("Cross-fold cosine design has invalid coverage.")
    return rows


def predicate_boundaries(h192: list[float]) -> list[dict[str, Any]]:
    vector = _finite_vector(h192, H192_WIDTH, "h192")
    jacobian = analytic_predicate_jacobian()
    predicates = matrix_vector(jacobian, vector)
    rows = []
    for group, offset in (("target_plus_1", 1.0), ("target", 0.0), ("target_minus_1", -1.0)):
        for index, (value, target, normal) in enumerate(
            zip(predicates, PREDICATE_TARGETS, jacobian)
        ):
            residual = value - (float(target) + offset)
            normal_norm = math.sqrt(sum(item * item for item in normal))
            rows.append(
                {
                    "row": len(rows),
                    "predicate_index": index,
                    "group": group,
                    "threshold": float(target) + offset,
                    "preactivation": residual,
                    "euclidean_distance": abs(residual) / normal_norm,
                    "on_boundary": abs(residual) <= BOUNDARY_TOLERANCE,
                }
            )
    return rows


def readout_gradient(h192: list[float]) -> list[float]:
    vector = _finite_vector(h192, H192_WIDTH, "h192")
    jacobian = analytic_predicate_jacobian()
    predicates = matrix_vector(jacobian, vector)
    gradient = [0.0] * H192_WIDTH
    for index, (value, target) in enumerate(zip(predicates, PREDICATE_TARGETS)):
        coefficient = (
            float(value > target + 1.0)
            - 2.0 * float(value > target)
            + float(value > target - 1.0)
        )
        for coordinate, item in enumerate(jacobian[index]):
            gradient[coordinate] += coefficient * item
    return gradient


def validate_geometry_report(document: Any) -> dict[str, Any]:
    required = {
        "schema_version",
        "analysis_kind",
        "source",
        "runtime",
        "directions",
        "predicate_geometry",
        "direction_geometry",
        "boundary_analysis",
        "curvature",
        "plots",
        "conclusion",
    }
    if not isinstance(document, dict) or set(document) != required:
        raise ProbeSchemaError("Local-geometry report has an invalid top-level field set.")
    if document["schema_version"] != GEOMETRY_SCHEMA_VERSION:
        raise ProbeSchemaError("Local-geometry report schema version is unsupported.")
    if document["analysis_kind"] != "relu_local_geometry":
        raise ProbeSchemaError("Local-geometry report kind is unsupported.")
    source = document["source"]
    if not isinstance(source, dict) or not all(
        _is_sha256(value) for key, value in source.items() if key.endswith("sha256")
    ):
        raise ProbeSchemaError("Local-geometry source hashes are invalid.")
    directions = document["directions"]
    if not isinstance(directions, list) or len(directions) != 30:
        raise ProbeSchemaError("Local-geometry report must contain thirty directions.")
    jacobian = document["predicate_geometry"].get("jacobian")
    if (
        not isinstance(jacobian, list)
        or len(jacobian) != PREDICATE_COUNT
        or any(not isinstance(row, list) or len(row) != H192_WIDTH for row in jacobian)
    ):
        raise ProbeSchemaError("Local-geometry predicate Jacobian shape is invalid.")
    direction_geometry = document.get("direction_geometry")
    model = (
        direction_geometry.get("cross_fold_cosine_model")
        if isinstance(direction_geometry, dict)
        else None
    )
    if not isinstance(model, dict):
        raise ProbeSchemaError("Local-geometry cross-fold cosine model is missing.")
    expected_counts = {
        "row_count": 150,
        "block_count": 6,
        "same_category_row_count": 30,
        "different_category_row_count": 120,
    }
    if any(model.get(key) != value for key, value in expected_counts.items()):
        raise ProbeSchemaError("Local-geometry cross-fold cosine model coverage is invalid.")
    if not isinstance(model.get("rows"), list) or len(model["rows"]) != 150:
        raise ProbeSchemaError("Local-geometry cross-fold cosine model rows are invalid.")
    permutation = model.get("permutation")
    if (
        not isinstance(permutation, dict)
        or type(permutation.get("repetitions")) is not int
        or permutation["repetitions"] <= 0
        or type(permutation.get("two_sided_p_value")) not in {int, float}
        or not 0.0 <= float(permutation["two_sided_p_value"]) <= 1.0
    ):
        raise ProbeSchemaError("Local-geometry cross-fold permutation result is invalid.")
    return document
