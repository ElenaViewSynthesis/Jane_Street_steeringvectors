from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "outputs" / "reports" / "architecture_report.json"


def linear_shapes(report: dict[str, Any]) -> list[tuple[str, int, int]]:
    """Return (module name, input width, output width) for every Linear module."""
    try:
        modules = report["loaded_object"]["module_tree"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Report does not contain loaded_object.module_tree.") from exc

    shapes: list[tuple[str, int, int]] = []
    for module in modules:
        if not module.get("type", "").endswith(".Linear"):
            continue
        try:
            weight = next(
                parameter
                for parameter in module["direct_parameters"]
                if parameter["name"] == "weight"
            )
            output_features, input_features = weight["shape"]
            shapes.append((module["name"], input_features, output_features))
        except (KeyError, TypeError, ValueError, StopIteration) as exc:
            raise ValueError(
                f"Linear module {module.get('name', '<unknown>')} has invalid weight metadata."
            ) from exc
    if not shapes:
        raise ValueError("Report does not contain any Linear modules.")
    return shapes


def print_summary(shapes: list[tuple[str, int, int]]) -> None:
    print("first 25:")
    for shape in shapes[:25]:
        print(shape)

    print("last 25:")
    for shape in shapes[-25:]:
        print(shape)

    print("common:")
    shape_counts = Counter((input_width, output_width) for _, input_width, output_width in shapes)
    for shape, count in shape_counts.most_common(30):
        print(shape, count)

    widths = {width for _, input_width, output_width in shapes for width in (input_width, output_width)}
    print("width range:", min(widths), max(widths))
    print("distinct widths:", len(widths))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Linear layer dimensions from an architecture report."
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"Architecture report path (default: {DEFAULT_REPORT})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError("Architecture report must be a JSON object.")
        print_summary(linear_shapes(report))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Unable to summarize {args.report}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
