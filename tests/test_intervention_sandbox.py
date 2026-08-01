from __future__ import annotations

import unittest
from pathlib import Path

from scripts.run_intervention_sandbox import build_intervention_unshare_command
from tests.test_activation_schema import MANIFEST_SHA256, MODEL_SHA256, REPETITIONS


class InterventionSandboxTests(unittest.TestCase):
    def test_command_requests_isolation_and_binds_intervention_spec(self) -> None:
        spec_sha256 = "c" * 64
        command = build_intervention_unshare_command(
            unshare_path="/usr/bin/unshare", sandbox_root=Path("/tmp/intervention/root"),
            venv_path=Path("/analysis/venv"), model_path=Path("/tmp/intervention/model.pt"),
            manifest_path=Path("/tmp/intervention/manifest.json"),
            spec_path=Path("/tmp/intervention/spec.json"),
            output_path=Path("/tmp/intervention/output/report.json"),
            source_url="https://example.test/model.pt", expected_sha256=MODEL_SHA256,
            expected_manifest_sha256=MANIFEST_SHA256, expected_spec_sha256=spec_sha256,
            repetitions=REPETITIONS, cpu_seconds=60, memory_bytes=4 * 1024**3,
            output_limit_bytes=1024**2,
        )

        for namespace_flag in ("--user", "--mount", "--net", "--pid", "--ipc", "--uts"):
            self.assertIn(namespace_flag, command)
        self.assertIn(spec_sha256, command)
        self.assertEqual(command[-1], "linux-userns-landlock-intervention-v1")


if __name__ == "__main__":
    unittest.main()
