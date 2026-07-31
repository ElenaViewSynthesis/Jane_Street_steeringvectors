from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path

from scripts.architecture_worker import (
    code_metadata,
    collect_module_callables,
    derive_final_predicate_circuit,
)
from scripts.run_architecture_sandbox import (
    SANDBOX_PROFILE,
    build_unshare_command,
    publish_report,
    validate_model_python_compatibility,
    validate_architecture_report,
)


EXPECTED_SHA256 = "a" * 64


def valid_report() -> dict:
    return {
        "schema_version": 1,
        "artifact": {"sha256": EXPECTED_SHA256},
        "sandbox": {"profile": SANDBOX_PROFILE},
        "loaded_object": {"kind": "nn.Module"},
        "callables": [],
        "final_predicate_circuit": {},
        "inference_executed": False,
    }


class SandboxLauncherTests(unittest.TestCase):
    def test_requires_exact_python_311_for_python_311_artifact(self) -> None:
        validate_model_python_compatibility("model_3_11.pt", (3, 11, 14))

        with self.assertRaisesRegex(Exception, "Python 3.11"):
            validate_model_python_compatibility("model_3_11.pt", (3, 12, 3))

    def test_rejects_pre_311_artifact_on_newer_python(self) -> None:
        with self.assertRaisesRegex(Exception, "pre-3.11"):
            validate_model_python_compatibility("model.pt", (3, 11, 14))

    def test_command_requests_all_required_namespaces(self) -> None:
        command = build_unshare_command(
            unshare_path="/usr/bin/unshare",
            sandbox_root=Path("/tmp/sandbox/root"),
            venv_path=Path("/workspace/.venv"),
            model_path=Path("/workspace/model_3_11.pt"),
            output_path=Path("/tmp/sandbox/output/architecture.json"),
            source_url="https://example.test/model_3_11.pt",
            expected_sha256=EXPECTED_SHA256,
            cpu_seconds=60,
            memory_bytes=4 * 1024**3,
            output_limit_bytes=16 * 1024**2,
        )

        for namespace_flag in ("--user", "--mount", "--net", "--pid", "--ipc", "--uts"):
            self.assertIn(namespace_flag, command)
        self.assertIn("--map-root-user", command)
        self.assertIn("--kill-child=SIGKILL", command)
        self.assertEqual(command[-1], SANDBOX_PROFILE)

    def test_report_validation_rejects_wrong_hash(self) -> None:
        report = valid_report()
        report["artifact"]["sha256"] = "b" * 64

        with self.assertRaisesRegex(ValueError, "hash"):
            validate_architecture_report(report, EXPECTED_SHA256)

    def test_report_validation_requires_no_inference(self) -> None:
        report = valid_report()
        report["inference_executed"] = True

        with self.assertRaisesRegex(ValueError, "inference"):
            validate_architecture_report(report, EXPECTED_SHA256)

    def test_publish_report_rejects_symbolic_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.json"
            target.write_text(json.dumps(valid_report()), encoding="utf-8")
            staged = root / "staged.json"
            staged.symlink_to(target)

            with self.assertRaisesRegex(ValueError, "regular file"):
                publish_report(staged, root / "result.json", EXPECTED_SHA256, 1_000_000)

    def test_publish_report_atomically_copies_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged = root / "staged.json"
            destination = root / "published" / "architecture.json"
            staged.write_text(json.dumps(valid_report()), encoding="utf-8")

            publish_report(staged, destination, EXPECTED_SHA256, 1_000_000)

            self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), valid_report())


class CallableInspectionTests(unittest.TestCase):
    def test_code_metadata_records_names_and_disassembly(self) -> None:
        def encode(text: str) -> list[int]:
            return [ord(character) for character in text]

        metadata = code_metadata(encode.__code__)

        self.assertEqual(metadata["name"], "encode")
        self.assertIn("ord", metadata["referenced_names"])
        self.assertIn("RETURN_VALUE", metadata["disassembly"])
        self.assertEqual(len(metadata["bytecode_sha256"]), 64)

    def test_collects_root_forward_and_registered_hook(self) -> None:
        def preprocess(_module, inputs):
            return inputs

        class FakeModule:
            def __init__(self) -> None:
                self._forward_pre_hooks = {0: preprocess}

            def forward(self, value):
                return value

            def named_modules(self):
                yield "", self

        callables = collect_module_callables(FakeModule())
        labels = [item["label"] for item in callables]

        self.assertIn("<root>.forward", labels)
        self.assertTrue(any("_forward_pre_hooks" in label for label in labels))


class FinalCircuitDerivationTests(unittest.TestCase):
    def test_derives_three_relu_equality_predicates(self) -> None:
        predicate_rows = [
            [1.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 2.0],
        ]
        predicate_layer = {
            "name": "predicate",
            "parameters": [
                {
                    "name": "weight",
                    "values": predicate_rows * 3,
                },
                {
                    "name": "bias",
                    "values": [-6.0, -8.0, -5.0, -7.0, -4.0, -6.0],
                },
            ],
        }
        output_layer = {
            "name": "output",
            "parameters": [
                {
                    "name": "weight",
                    "values": [[1.0, 1.0, -2.0, -2.0, 1.0, 1.0]],
                },
                {"name": "bias", "values": [-1.0]},
            ],
        }

        circuit = derive_final_predicate_circuit(
            [predicate_layer, output_layer],
            block_width=2,
        )

        self.assertEqual(circuit["predicate_count"], 2)
        self.assertEqual(
            circuit["predicates"],
            [
                {
                    "index": 0,
                    "target": 5.0,
                    "block_terms": [{"block": 0, "coefficient": 1.0}],
                    "relu_biases": [-6.0, -5.0, -4.0],
                },
                {
                    "index": 1,
                    "target": 7.0,
                    "block_terms": [{"block": 1, "coefficient": 1.0}],
                    "relu_biases": [-8.0, -7.0, -6.0],
                },
            ],
        )
        self.assertEqual(circuit["indicator_weights"], [1.0, -2.0, 1.0])
        self.assertEqual(circuit["final_bias"], -1.0)


if __name__ == "__main__":
    unittest.main()
