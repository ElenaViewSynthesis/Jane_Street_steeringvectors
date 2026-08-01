from __future__ import annotations

import copy
import unittest
from pathlib import Path

from scripts.activation_schema import (
    CAPTURE_MODULES,
    CAPTURE_PROFILE,
    PREDICATE_TARGETS,
    SANDBOX_PROFILE,
    decode_q_values,
    derive_activation_values,
    digest_candidates,
    float32_sha256,
    summarize_activation_results,
    validate_activation_report,
)
from scripts.probe_schema import ProbeSchemaError, load_probe_manifest
from tests.test_probe_schema import valid_manifest


MODEL_SHA256 = "a" * 64
MANIFEST_SHA256 = "b" * 64
REPETITIONS = 2


def tensor_snapshot(values: list[float], shape: list[int]) -> dict:
    return {
        "dtype": "torch.float32",
        "shape": shape,
        "device": "cpu",
        "values": values,
        "float32_sha256": float32_sha256(values),
    }


def valid_tensors() -> dict:
    a48 = (
        [-(target + 1.0) for target in PREDICATE_TARGETS]
        + [-target for target in PREDICATE_TARGETS]
        + [-(target - 1.0) for target in PREDICATE_TARGETS]
    )
    return {
        "h192": tensor_snapshot([0.0] * 192, [192]),
        "a48": tensor_snapshot(a48, [48]),
        "z48": tensor_snapshot([0.0] * 48, [48]),
        "readout": tensor_snapshot([-15.0], [1]),
        "output": tensor_snapshot([0.0], [1]),
    }


def valid_observation(input_text: str, repetition: int) -> dict:
    tensors = valid_tensors()
    return {
        "repetition": repetition,
        "tensors": tensors,
        "derived": derive_activation_values(input_text, tensors),
        "verification": {
            "a48_max_abs_error": 0.0,
            "z48_max_abs_error": 0.0,
            "readout_max_abs_error": 0.0,
            "output_max_abs_error": 0.0,
            "returned_output_matches_hook": True,
        },
    }


def valid_activation_report() -> dict:
    manifest = valid_manifest()
    results = []
    for case in manifest["cases"]:
        observations = [
            valid_observation(case["input"], repetition)
            for repetition in range(REPETITIONS)
        ]
        results.append(
            {
                "case": case,
                "input_length": len(case["input"]),
                "observations": observations,
                "deterministic": True,
            }
        )
    return {
        "schema_version": 1,
        "artifact": {
            "filename": "model_3_11.pt",
            "size_bytes": 1_158_729_818,
            "sha256": MODEL_SHA256,
            "source_url": "https://example.test/model_3_11.pt",
        },
        "probe_suite": {
            "suite_id": manifest["suite_id"],
            "description": manifest["description"],
            "manifest_filename": "manifest.json",
            "manifest_sha256": MANIFEST_SHA256,
            "case_count": len(manifest["cases"]),
            "repetitions": REPETITIONS,
        },
        "capture": {
            "profile": CAPTURE_PROFILE,
            "scoped_hooks": True,
            "hooks_removed": True,
            "modules": CAPTURE_MODULES,
        },
        "sandbox": {
            "profile": SANDBOX_PROFILE,
            "network_namespace": "isolated",
            "filesystem_policy": "Landlock allow-list",
            "model_access": "read-only",
            "manifest_access": "read-only",
            "host_credentials": "not allow-listed",
            "output_access": "staging-directory-only",
            "capabilities": "dropped",
            "no_new_privileges": True,
        },
        "runtime": {
            "python": "3.11.14",
            "torch": "2.7.1+cpu",
            "cloudpickle": "3.1.1",
            "platform": "Linux-test",
        },
        "execution": {
            "inference_mode": True,
            "model_eval": True,
            "deterministic_algorithms": True,
            "torch_num_threads": 1,
            "torch_num_interop_threads": 1,
            "seed": 0,
        },
        "results": results,
        "summary": summarize_activation_results(results),
        "inference_executed": True,
    }


def validate(report: dict) -> dict:
    return validate_activation_report(
        report,
        expected_sha256=MODEL_SHA256,
        expected_manifest_sha256=MANIFEST_SHA256,
        manifest=valid_manifest(),
        repetitions=REPETITIONS,
    )


class ActivationDerivationTests(unittest.TestCase):
    def test_decodes_little_endian_bit_blocks(self) -> None:
        h192 = [0.0] * 192
        h192[0] = 1.0
        h192[7] = 1.0
        h192[8 + 3] = 1.0

        q_values = decode_q_values(h192)

        self.assertEqual(q_values[:2], [129.0, 8.0])
        self.assertEqual(q_values[2:], [0.0] * 22)

    def test_compares_decoded_bytes_with_md5_candidates(self) -> None:
        ordinary_md5 = bytes.fromhex("900150983cd24fb0d6963f7d28e17f72")
        candidates = digest_candidates("abc", [float(value) for value in ordinary_md5])

        self.assertTrue(candidates["decoded_matches_utf8"])
        self.assertTrue(candidates["decoded_matches_truncated_utf8"])
        self.assertFalse(candidates["decoded_matches_padded_codepoint_bytes"])


class ActivationReportValidationTests(unittest.TestCase):
    def test_accepts_complete_activation_report(self) -> None:
        report = valid_activation_report()
        self.assertIs(validate(report), report)

    def test_rejects_changed_tensor_values_without_digest_update(self) -> None:
        report = valid_activation_report()
        report["results"][0]["observations"][0]["tensors"]["h192"]["values"][0] = 1.0

        with self.assertRaisesRegex(ProbeSchemaError, "digest"):
            validate(report)

    def test_rejects_inconsistent_derived_values(self) -> None:
        report = valid_activation_report()
        report["results"][0]["observations"][0]["derived"][
            "matched_predicate_count"
        ] = 16

        with self.assertRaisesRegex(ProbeSchemaError, "derived"):
            validate(report)

    def test_rejects_algebraically_inconsistent_stages(self) -> None:
        report = valid_activation_report()
        snapshot = report["results"][0]["observations"][0]["tensors"]["z48"]
        snapshot["values"][0] = 1.0
        snapshot["float32_sha256"] = float32_sha256(snapshot["values"])

        with self.assertRaisesRegex(ProbeSchemaError, "algebraically inconsistent"):
            validate(report)

    def test_rejects_affine_error_above_tolerance(self) -> None:
        report = valid_activation_report()
        report["results"][0]["observations"][0]["verification"][
            "readout_max_abs_error"
        ] = 0.1

        with self.assertRaisesRegex(ProbeSchemaError, "exceeds tolerance"):
            validate(report)

    def test_rejects_report_claiming_hooks_remained_installed(self) -> None:
        report = copy.deepcopy(valid_activation_report())
        report["capture"]["hooks_removed"] = False

        with self.assertRaisesRegex(ProbeSchemaError, "capture metadata"):
            validate(report)

    def test_committed_smoke_manifest_is_valid(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest, _payload, digest = load_probe_manifest(
            root / "experiments" / "activations" / "m4-capture-smoke-v1.json"
        )

        self.assertEqual(manifest["suite_id"], "m4-capture-smoke-v1")
        self.assertEqual(len(manifest["cases"]), 4)
        self.assertEqual(len(digest), 64)

    def test_committed_boundary_manifest_is_valid(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest, _payload, digest = load_probe_manifest(
            root / "experiments" / "activations" / "m4-md5-boundary-v1.json"
        )

        self.assertEqual(manifest["suite_id"], "m4-md5-boundary-v1")
        self.assertEqual(len(manifest["cases"]), 10)
        self.assertEqual(len(digest), 64)


if __name__ == "__main__":
    unittest.main()
