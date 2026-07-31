from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.probe_schema import ProbeSchemaError, summarize_probe_results
from scripts.run_probe_sandbox import (
    SANDBOX_PROFILE,
    build_probe_unshare_command,
    publish_probe_report,
    validate_probe_report,
)
from tests.test_probe_schema import valid_manifest


MODEL_SHA256 = "a" * 64
MANIFEST_SHA256 = "b" * 64
REPETITIONS = 2


def valid_report() -> dict:
    manifest = valid_manifest()
    results = []
    for case in manifest["cases"]:
        observations = [
            {
                "repetition": repetition,
                "value": 0.0,
                "float_hex": 0.0.hex(),
                "dtype": "torch.float32",
                "shape": [1],
                "device": "cpu",
            }
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
        "summary": summarize_probe_results(results),
        "inference_executed": True,
    }


def validate(report: dict) -> dict:
    return validate_probe_report(
        report,
        expected_sha256=MODEL_SHA256,
        expected_manifest_sha256=MANIFEST_SHA256,
        manifest=valid_manifest(),
        repetitions=REPETITIONS,
    )


class ProbeSandboxTests(unittest.TestCase):
    def test_command_requests_namespaces_and_binds_manifest(self) -> None:
        command = build_probe_unshare_command(
            unshare_path="/usr/bin/unshare",
            sandbox_root=Path("/tmp/probe/root"),
            venv_path=Path("/analysis/venv"),
            model_path=Path("/tmp/probe/model_3_11.pt"),
            manifest_path=Path("/tmp/probe/manifest.json"),
            output_path=Path("/tmp/probe/output/report.json"),
            source_url="https://example.test/model.pt",
            expected_sha256=MODEL_SHA256,
            expected_manifest_sha256=MANIFEST_SHA256,
            repetitions=REPETITIONS,
            cpu_seconds=60,
            memory_bytes=4 * 1024**3,
            output_limit_bytes=1024**2,
        )

        for namespace_flag in ("--user", "--mount", "--net", "--pid", "--ipc", "--uts"):
            self.assertIn(namespace_flag, command)
        self.assertIn(str(Path("/tmp/probe/manifest.json")), command)
        self.assertIn(MANIFEST_SHA256, command)
        self.assertEqual(command[-1], SANDBOX_PROFILE)

    def test_accepts_a_complete_deterministic_report(self) -> None:
        report = valid_report()
        self.assertIs(validate(report), report)

    def test_rejects_wrong_model_hash(self) -> None:
        report = valid_report()
        report["artifact"]["sha256"] = "c" * 64
        with self.assertRaisesRegex(ProbeSchemaError, "artifact hash"):
            validate(report)

    def test_rejects_case_input_not_bound_to_manifest(self) -> None:
        report = valid_report()
        report["results"][0]["case"]["input"] = "changed"
        with self.assertRaisesRegex(ProbeSchemaError, "manifest case"):
            validate(report)

    def test_rejects_inconsistent_summary(self) -> None:
        report = valid_report()
        report["summary"]["unique_output_count"] = 99
        with self.assertRaisesRegex(ProbeSchemaError, "summary"):
            validate(report)

    def test_rejects_inconsistent_float_encoding(self) -> None:
        report = valid_report()
        report["results"][0]["observations"][0]["float_hex"] = 1.0.hex()
        with self.assertRaisesRegex(ProbeSchemaError, "float encoding"):
            validate(report)

    def test_publish_rejects_symbolic_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.json"
            target.write_text(json.dumps(valid_report()), encoding="utf-8")
            staged = root / "staged.json"
            staged.symlink_to(target)

            with self.assertRaisesRegex(ProbeSchemaError, "regular file"):
                publish_probe_report(
                    staged,
                    root / "published.json",
                    maximum_bytes=1_000_000,
                    expected_sha256=MODEL_SHA256,
                    expected_manifest_sha256=MANIFEST_SHA256,
                    manifest=valid_manifest(),
                    repetitions=REPETITIONS,
                )

    def test_publish_atomically_copies_valid_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged = root / "staged.json"
            destination = root / "published" / "probe.json"
            report = valid_report()
            staged.write_text(json.dumps(report), encoding="utf-8")

            published = publish_probe_report(
                staged,
                destination,
                maximum_bytes=1_000_000,
                expected_sha256=MODEL_SHA256,
                expected_manifest_sha256=MANIFEST_SHA256,
                manifest=valid_manifest(),
                repetitions=REPETITIONS,
            )
            self.assertEqual(published, report)
            self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), report)


if __name__ == "__main__":
    unittest.main()
