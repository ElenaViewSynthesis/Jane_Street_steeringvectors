from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.activation_schema import SANDBOX_PROFILE
from scripts.probe_schema import ProbeSchemaError
from scripts.run_activation_sandbox import (
    build_activation_unshare_command,
    publish_activation_report,
)
from tests.test_activation_schema import (
    MANIFEST_SHA256,
    MODEL_SHA256,
    REPETITIONS,
    valid_activation_report,
)
from tests.test_probe_schema import valid_manifest


class ActivationSandboxTests(unittest.TestCase):
    def test_command_requests_namespaces_and_binds_manifest(self) -> None:
        command = build_activation_unshare_command(
            unshare_path="/usr/bin/unshare",
            sandbox_root=Path("/tmp/activation/root"),
            venv_path=Path("/analysis/venv"),
            model_path=Path("/tmp/activation/model_3_11.pt"),
            manifest_path=Path("/tmp/activation/manifest.json"),
            output_path=Path("/tmp/activation/output/report.json"),
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
        self.assertIn(MANIFEST_SHA256, command)
        self.assertEqual(command[-1], SANDBOX_PROFILE)

    def test_publish_rejects_symbolic_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.json"
            target.write_text(json.dumps(valid_activation_report()), encoding="utf-8")
            staged = root / "staged.json"
            staged.symlink_to(target)

            with self.assertRaisesRegex(ProbeSchemaError, "regular file"):
                publish_activation_report(
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
            destination = root / "published" / "activation.json"
            report = valid_activation_report()
            staged.write_text(json.dumps(report), encoding="utf-8")

            published = publish_activation_report(
                staged,
                destination,
                maximum_bytes=2_000_000,
                expected_sha256=MODEL_SHA256,
                expected_manifest_sha256=MANIFEST_SHA256,
                manifest=valid_manifest(),
                repetitions=REPETITIONS,
            )

            self.assertEqual(published, report)
            self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), report)


if __name__ == "__main__":
    unittest.main()
