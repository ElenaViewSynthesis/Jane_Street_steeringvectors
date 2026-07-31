from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.generate_probe_manifests import factorial_manifest, smoke_manifest
from scripts.probe_schema import (
    ProbeSchemaError,
    load_probe_manifest,
    loads_json,
    validate_probe_manifest,
)


def valid_manifest() -> dict:
    return {
        "schema_version": 1,
        "suite_id": "test-suite-v1",
        "description": "A deterministic test suite.",
        "cases": [
            {
                "id": "case-a",
                "input": "alpha",
                "family": "baseline",
                "factors": {"variant": "plain", "enabled": True},
            },
            {
                "id": "case-b",
                "input": "beta",
                "family": "baseline",
                "factors": {"variant": "plain", "enabled": False},
            },
        ],
    }


class ProbeManifestTests(unittest.TestCase):
    def test_accepts_and_loads_a_versioned_manifest(self) -> None:
        manifest = valid_manifest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            payload = (json.dumps(manifest, indent=2) + "\n").encode()
            path.write_bytes(payload)

            loaded, loaded_payload, digest = load_probe_manifest(path)

        self.assertEqual(loaded, manifest)
        self.assertEqual(loaded_payload, payload)
        self.assertEqual(len(digest), 64)

    def test_rejects_duplicate_json_keys(self) -> None:
        with self.assertRaisesRegex(ProbeSchemaError, "duplicate key"):
            loads_json('{"schema_version": 1, "schema_version": 1}')

    def test_rejects_duplicate_case_ids(self) -> None:
        manifest = valid_manifest()
        manifest["cases"][1]["id"] = manifest["cases"][0]["id"]

        with self.assertRaisesRegex(ProbeSchemaError, "Duplicate"):
            validate_probe_manifest(manifest)

    def test_rejects_non_string_input(self) -> None:
        manifest = valid_manifest()
        manifest["cases"][0]["input"] = ["not", "text"]

        with self.assertRaisesRegex(ProbeSchemaError, "input must be a string"):
            validate_probe_manifest(manifest)

    def test_rejects_nested_factor_values(self) -> None:
        manifest = valid_manifest()
        manifest["cases"][0]["factors"]["nested"] = {"value": 1}

        with self.assertRaisesRegex(ProbeSchemaError, "must be a string"):
            validate_probe_manifest(manifest)


class GeneratedProbeSuiteTests(unittest.TestCase):
    def test_smoke_suite_covers_required_contrasts(self) -> None:
        manifest = smoke_manifest()
        families = {case["family"] for case in manifest["cases"]}
        transforms = {
            case["factors"].get("transform")
            for case in manifest["cases"]
            if case["family"] == "controlled_contrast"
        }

        self.assertEqual(len(manifest["cases"]), 32)
        self.assertEqual(
            families,
            {"baseline", "padding", "length", "controlled_contrast"},
        )
        self.assertTrue(
            {"original", "reversed", "uppercase", "titlecase", "punctuation", "repeated"}
            <= transforms
        )

    def test_smoke_suite_contains_encoding_equivalence_pairs(self) -> None:
        cases = {case["id"]: case["input"] for case in smoke_manifest()["cases"]}
        encode = lambda value: value[:55].ljust(55, "\x00")

        self.assertEqual(
            encode(cases["padding-short-abc"]),
            encode(cases["padding-explicit-abc"]),
        )
        self.assertEqual(
            encode(cases["length-a-55"]),
            encode(cases["length-a-56"]),
        )

    def test_factorial_suite_is_complete_and_balanced(self) -> None:
        manifest = factorial_manifest()
        cases = manifest["cases"]
        categories = {
            case["factors"][side]
            for case in cases
            for side in ("left_category", "right_category")
        }

        self.assertEqual(len(cases), 225)
        self.assertEqual(categories, {"animal", "food", "color", "place", "verb"})
        self.assertEqual(len({case["id"] for case in cases}), 225)

    def test_committed_manifests_match_the_generator(self) -> None:
        root = Path(__file__).resolve().parents[1]
        expected = {
            "m3-smoke-v1.json": smoke_manifest(),
            "m3-semantic-factorial-v1.json": factorial_manifest(),
        }
        for filename, generated in expected.items():
            committed, _payload, _digest = load_probe_manifest(
                root / "experiments" / "probes" / filename
            )
            self.assertEqual(committed, generated)


if __name__ == "__main__":
    unittest.main()
