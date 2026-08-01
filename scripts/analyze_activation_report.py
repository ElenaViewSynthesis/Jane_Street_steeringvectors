from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    from scripts.activation_schema import validate_activation_report
    from scripts.probe_schema import (
        MAX_MANIFEST_BYTES,
        ProbeSchemaError,
        load_probe_manifest,
        loads_json,
        read_bounded_regular_file,
        sha256_bytes,
    )
except ModuleNotFoundError:  # Direct execution from scripts/.
    from activation_schema import validate_activation_report  # type: ignore[no-redef]
    from probe_schema import (  # type: ignore[no-redef]
        MAX_MANIFEST_BYTES,
        ProbeSchemaError,
        load_probe_manifest,
        loads_json,
        read_bounded_regular_file,
        sha256_bytes,
    )


ANALYSIS_SCHEMA_VERSION = 1
MAX_REPORT_BYTES = 64 * 1024 * 1024
RANDOMIZATION_REPETITIONS = 64


def load_validated_activation_report(
    report_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], str]:
    manifest, _manifest_payload, manifest_sha256 = load_probe_manifest(manifest_path)
    payload = read_bounded_regular_file(report_path, MAX_REPORT_BYTES)
    report = loads_json(payload)
    if not isinstance(report, dict) or not isinstance(report.get("artifact"), dict):
        raise ProbeSchemaError("Activation report is missing artifact metadata.")
    validated = validate_activation_report(
        report,
        expected_sha256=report["artifact"].get("sha256"),
        expected_manifest_sha256=manifest_sha256,
        manifest=manifest,
        repetitions=report.get("probe_suite", {}).get("repetitions"),
    )
    return validated, sha256_bytes(payload)


def _first_observation(result: dict[str, Any]) -> dict[str, Any]:
    observations = result["observations"]
    if not observations:
        raise ProbeSchemaError("Activation result has no observations.")
    return observations[0]


def analyze_encoding(report: dict[str, Any], report_sha256: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    candidate_names: set[str] = set()
    for result in report["results"]:
        observation = _first_observation(result)
        candidates = observation["derived"]["digest_candidates"]
        match_fields = {
            key: value
            for key, value in candidates.items()
            if key.startswith("decoded_matches_")
        }
        candidate_names.update(match_fields)
        rows.append(
            {
                "case_id": result["case"]["id"],
                "input": result["case"]["input"],
                "factors": result["case"]["factors"],
                "decoded_predicate_hex": candidates["decoded_predicate_hex"],
                "matches": match_fields,
            }
        )
    candidate_summary = [
        {
            "candidate": name,
            "match_count": sum(int(row["matches"].get(name, False)) for row in rows),
            "available_count": sum(name in row["matches"] for row in rows),
            "matches_all_cases": all(row["matches"].get(name) is True for row in rows),
        }
        for name in sorted(candidate_names)
    ]
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_kind": "md5_encoding_boundary",
        "source": {
            "activation_report_sha256": report_sha256,
            "artifact_sha256": report["artifact"]["sha256"],
            "manifest_sha256": report["probe_suite"]["manifest_sha256"],
            "suite_id": report["probe_suite"]["suite_id"],
        },
        "cases": rows,
        "candidate_summary": candidate_summary,
        "conclusion": {
            "best_candidates": [
                item["candidate"] for item in candidate_summary if item["matches_all_cases"]
            ],
            "rule_status": "candidate_match_only",
            "caveat": (
                "A matching candidate identifies behavior only for this manifest; additional "
                "boundary cases are required before treating it as a universal encoding rule."
            ),
        },
    }


def _mean(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise ProbeSchemaError("Cannot average an empty vector collection.")
    width = len(vectors[0])
    if any(len(vector) != width for vector in vectors):
        raise ProbeSchemaError("Activation vectors have inconsistent widths.")
    return [sum(vector[index] for vector in vectors) / len(vectors) for index in range(width)]


def _subtract(left: list[float], right: list[float]) -> list[float]:
    return [a - b for a, b in zip(left, right)]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(_dot(vector, vector))
    return [value / norm for value in vector] if norm else [0.0] * len(vector)


def _argmax_label(scores: dict[str, float]) -> str:
    return max(sorted(scores), key=lambda label: scores[label])


def _semantic_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for result in report["results"]:
        case = result["case"]
        factors = case["factors"]
        required = {"left_word", "left_category", "right_word", "right_category"}
        if not required <= set(factors):
            raise ProbeSchemaError("Semantic analysis requires the factorial manifest factors.")
        h192 = _first_observation(result)["tensors"]["h192"]["values"]
        rows.append(
            {
                "case_id": case["id"],
                "left_word": factors["left_word"],
                "left_category": factors["left_category"],
                "right_word": factors["right_word"],
                "right_category": factors["right_category"],
                "h192": h192,
            }
        )
    return rows


def _fit_category_directions(
    rows: list[dict[str, Any]],
    *,
    slot: str,
    categories: list[str],
    labels: list[str] | None = None,
) -> dict[str, list[float]]:
    category_key = f"{slot}_category"
    effective_labels = labels or [row[category_key] for row in rows]
    if len(effective_labels) != len(rows):
        raise ProbeSchemaError("Semantic labels do not align with the training rows.")
    directions: dict[str, list[float]] = {}
    for category in categories:
        positive = [
            row["h192"] for row, label in zip(rows, effective_labels) if label == category
        ]
        negative = [
            row["h192"] for row, label in zip(rows, effective_labels) if label != category
        ]
        directions[category] = _l2_normalize(_subtract(_mean(positive), _mean(negative)))
    return directions


def _evaluate_category_directions(
    rows: list[dict[str, Any]],
    directions: dict[str, list[float]],
    *,
    slot: str,
) -> dict[str, Any]:
    category_key = f"{slot}_category"
    predictions = []
    for row in rows:
        scores = {category: _dot(row["h192"], vector) for category, vector in directions.items()}
        predicted = _argmax_label(scores)
        predictions.append(
            {
                "case_id": row["case_id"],
                "actual": row[category_key],
                "predicted": predicted,
                "correct": predicted == row[category_key],
            }
        )
    return {
        "case_count": len(predictions),
        "correct_count": sum(item["correct"] for item in predictions),
        "accuracy": sum(item["correct"] for item in predictions) / len(predictions),
        "predictions": predictions,
    }


def _randomized_accuracy(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    slot: str,
    categories: list[str],
    seed: int,
) -> list[float]:
    labels = [row[f"{slot}_category"] for row in train_rows]
    generator = random.Random(seed)
    accuracies = []
    for _ in range(RANDOMIZATION_REPETITIONS):
        shuffled = list(labels)
        generator.shuffle(shuffled)
        directions = _fit_category_directions(
            train_rows,
            slot=slot,
            categories=categories,
            labels=shuffled,
        )
        accuracies.append(_evaluate_category_directions(test_rows, directions, slot=slot)["accuracy"])
    return accuracies


def _leaky_word_identity_baseline(rows: list[dict[str, Any]], slot: str) -> dict[str, Any]:
    word_key = f"{slot}_word"
    # A case-id hash makes this intentionally leaky diagnostic independent of manifest order,
    # while retaining every lexeme in the training set for both word slots.
    def is_test(row: dict[str, Any]) -> bool:
        digest = hashlib.sha256(row["case_id"].encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % 5 == 0

    train_rows = [row for row in rows if not is_test(row)]
    test_rows = [row for row in rows if is_test(row)]
    words = sorted({row[word_key] for row in train_rows})
    centroids = {
        word: _mean([row["h192"] for row in train_rows if row[word_key] == word])
        for word in words
    }
    eligible = [row for row in test_rows if row[word_key] in centroids]
    correct = 0
    for row in eligible:
        predicted = min(
            words,
            key=lambda word: sum(
                (value - center) ** 2 for value, center in zip(row["h192"], centroids[word])
            ),
        )
        correct += int(predicted == row[word_key])
    return {
        "split": "deterministic_case_id_hash_80_20_leaky",
        "eligible_case_count": len(eligible),
        "accuracy": correct / len(eligible) if eligible else None,
        "interpretation": (
            "This intentionally leaky word-identity score is a diagnostic only; it must not be "
            "used as evidence of held-out semantic generalization."
        ),
    }


def analyze_semantic(report: dict[str, Any], report_sha256: str) -> dict[str, Any]:
    rows = _semantic_rows(report)
    categories = sorted({row["left_category"] for row in rows})
    words_by_category = {
        category: sorted(
            {row["left_word"] for row in rows if row["left_category"] == category}
        )
        for category in categories
    }
    if any(len(words) < 2 for words in words_by_category.values()):
        raise ProbeSchemaError("Each semantic category needs at least two distinct lexemes.")
    fold_count = min(len(words) for words in words_by_category.values())
    slot_results: dict[str, Any] = {}
    for slot in ("left", "right"):
        folds = []
        for fold_index in range(fold_count):
            held_out = {
                words_by_category[category][fold_index] for category in categories
            }
            train_rows = [
                row
                for row in rows
                if row["left_word"] not in held_out and row["right_word"] not in held_out
            ]
            test_rows = [
                row
                for row in rows
                if row[f"{slot}_word"] in held_out
                and row[f"{'right' if slot == 'left' else 'left'}_word"] not in held_out
            ]
            directions = _fit_category_directions(
                train_rows,
                slot=slot,
                categories=categories,
            )
            evaluation = _evaluate_category_directions(test_rows, directions, slot=slot)
            random_accuracies = _randomized_accuracy(
                train_rows,
                test_rows,
                slot=slot,
                categories=categories,
                seed=fold_index + (0 if slot == "left" else 10_000),
            )
            folds.append(
                {
                    "fold_index": fold_index,
                    "held_out_words": sorted(held_out),
                    "train_case_count": len(train_rows),
                    "test_case_count": len(test_rows),
                    "directions": directions,
                    "evaluation": evaluation,
                    "randomized_label_control": {
                        "repetitions": RANDOMIZATION_REPETITIONS,
                        "mean_accuracy": sum(random_accuracies) / len(random_accuracies),
                        "maximum_accuracy": max(random_accuracies),
                        "accuracies": random_accuracies,
                    },
                }
            )
        accuracies = [fold["evaluation"]["accuracy"] for fold in folds]
        slot_results[slot] = {
            "folds": folds,
            "mean_held_out_accuracy": sum(accuracies) / len(accuracies),
            "chance_accuracy": 1.0 / len(categories),
            "leaky_word_identity_baseline": _leaky_word_identity_baseline(rows, slot),
        }
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_kind": "held_out_semantic_directions",
        "source": {
            "activation_report_sha256": report_sha256,
            "artifact_sha256": report["artifact"]["sha256"],
            "manifest_sha256": report["probe_suite"]["manifest_sha256"],
            "suite_id": report["probe_suite"]["suite_id"],
        },
        "protocol": {
            "split": "leave_one_lexeme_per_category_out_across_three_folds",
            "training_filter": "exclude every case containing any held-out lexeme",
            "direction": "normalized one-vs-rest mean difference in h192",
            "randomized_label_control_repetitions": RANDOMIZATION_REPETITIONS,
        },
        "categories": categories,
        "words_by_category": words_by_category,
        "slots": slot_results,
        "interpretation": (
            "Held-out accuracy must be compared with chance and the randomized-label control. "
            "The leaky word-identity baseline is included only to expose why pair-level random "
            "splits are not valid semantic evidence."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze validated Milestone 4 activation reports."
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--kind", choices=("encoding", "semantic"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report, report_sha256 = load_validated_activation_report(args.report, args.manifest)
        analysis = (
            analyze_encoding(report, report_sha256)
            if args.kind == "encoding"
            else analyze_semantic(report, report_sha256)
        )
    except (OSError, ProbeSchemaError, RecursionError) as exc:
        print(f"Activation analysis rejected input: {exc}", file=sys.stderr)
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
        json.dump(analysis, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary_path, args.output)
    print(f"Activation analysis written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
