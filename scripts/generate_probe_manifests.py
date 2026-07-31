from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts.probe_schema import validate_probe_manifest
except ModuleNotFoundError:  # Direct execution from scripts/.
    from probe_schema import validate_probe_manifest  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRECTORY = ROOT / "experiments" / "probes"


def probe_case(
    case_id: str,
    input_text: str,
    family: str,
    **factors: str | int | bool | None,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "input": input_text,
        "family": family,
        "factors": factors,
    }


def smoke_manifest() -> dict[str, Any]:
    cases = [
        probe_case("baseline-empty", "", "baseline", variant="empty"),
        probe_case("baseline-single-a", "a", "baseline", variant="single_character"),
        probe_case("baseline-space", " ", "baseline", variant="space"),
        probe_case(
            "baseline-puzzle-example",
            "vegetable dog",
            "baseline",
            variant="documented_example",
        ),
        probe_case("padding-short-abc", "abc", "padding", variant="implicit_nulls"),
        probe_case(
            "padding-explicit-abc",
            "abc" + "\x00" * 52,
            "padding",
            variant="explicit_nulls",
        ),
        probe_case("length-a-54", "a" * 54, "length", character="a", length=54),
        probe_case("length-a-55", "a" * 55, "length", character="a", length=55),
        probe_case("length-a-56", "a" * 56, "length", character="a", length=56),
        probe_case(
            "truncation-a55-exclamation",
            "a" * 55 + "!",
            "length",
            character="a",
            length=56,
            suffix="exclamation",
        ),
        probe_case(
            "truncation-a55-question",
            "a" * 55 + "?",
            "length",
            character="a",
            length=56,
            suffix="question",
        ),
    ]

    contrast_bases = [
        ("red-fox", "red", "color", "fox", "animal"),
        ("apple-run", "apple", "food", "run", "verb"),
        ("paris-blue", "paris", "place", "blue", "color"),
    ]
    for base_id, left, left_category, right, right_category in contrast_bases:
        variants = [
            ("original", f"{left} {right}"),
            ("reversed", f"{right} {left}"),
            ("uppercase", f"{left} {right}".upper()),
            ("titlecase", f"{left} {right}".title()),
            ("punctuation", f"{left} {right}!"),
            ("repeated", f"{left} {right} {left} {right}"),
            ("double-space", f"{left}  {right}"),
        ]
        for transform, input_text in variants:
            cases.append(
                probe_case(
                    f"contrast-{base_id}-{transform}",
                    input_text,
                    "controlled_contrast",
                    base_id=base_id,
                    transform=transform,
                    left_category=left_category,
                    right_category=right_category,
                )
            )

    return validate_probe_manifest(
        {
            "schema_version": 1,
            "suite_id": "m3-smoke-v1",
            "description": (
                "Milestone 3 smoke suite covering deterministic baselines, input padding, "
                "the 55-character boundary, and controlled order, case, punctuation, "
                "repetition, and whitespace contrasts."
            ),
            "cases": cases,
        }
    )


def factorial_manifest() -> dict[str, Any]:
    vocabulary = {
        "animal": ("cat", "dog", "fox"),
        "food": ("apple", "bread", "carrot"),
        "color": ("blue", "green", "red"),
        "place": ("london", "paris", "rome"),
        "verb": ("jump", "run", "sleep"),
    }
    words = [
        (word, category)
        for category, category_words in vocabulary.items()
        for word in category_words
    ]
    cases = []
    for left_word, left_category in words:
        for right_word, right_category in words:
            cases.append(
                probe_case(
                    f"pair-{left_word}-{right_word}",
                    f"{left_word} {right_word}",
                    "semantic_factorial",
                    left_word=left_word,
                    left_category=left_category,
                    right_word=right_word,
                    right_category=right_category,
                    same_word=left_word == right_word,
                    same_category=left_category == right_category,
                )
            )
    return validate_probe_manifest(
        {
            "schema_version": 1,
            "suite_id": "m3-semantic-factorial-v1",
            "description": (
                "Complete ordered 15-by-15 word-pair factorial over animals, foods, "
                "colors, places, and verbs for lexical, positional, and interaction effects."
            ),
            "cases": cases,
        }
    )


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic Milestone 3 probes.")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifests = (smoke_manifest(), factorial_manifest())
    for manifest in manifests:
        path = args.output_directory / f"{manifest['suite_id']}.json"
        write_manifest(path, manifest)
        print(f"Wrote {len(manifest['cases'])} cases to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
