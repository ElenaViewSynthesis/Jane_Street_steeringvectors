from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.landlock_exec import landlock_abi_version


ROOT = Path(__file__).resolve().parents[1]
LANDLOCK_EXEC = ROOT / "scripts" / "landlock_exec.py"


class LandlockPolicyTests(unittest.TestCase):
    def test_kernel_exposes_required_landlock_abi(self) -> None:
        self.assertGreaterEqual(landlock_abi_version(), 3)

    def test_allows_explicitly_listed_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            allowed_file = Path(directory) / "allowed.txt"
            allowed_file.write_text("allowed", encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(LANDLOCK_EXEC),
                    "--allow-read",
                    "/usr",
                    "--allow-read",
                    "/etc/ld.so.cache",
                    "--allow-read",
                    str(allowed_file),
                    "--",
                    sys.executable,
                    "-I",
                    "-c",
                    "import pathlib,sys; print(pathlib.Path(sys.argv[1]).read_text())",
                    str(allowed_file),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "allowed")

    def test_denies_file_outside_allow_list(self) -> None:
        with tempfile.TemporaryDirectory() as allowed_directory:
            with tempfile.TemporaryDirectory() as denied_directory:
                denied_file = Path(denied_directory) / "denied.txt"
                denied_file.write_text("secret", encoding="utf-8")
                probe = (
                    "import pathlib,sys; "
                    "path=pathlib.Path(sys.argv[1]); "
                    "\ntry: path.read_text()"
                    "\nexcept PermissionError: raise SystemExit(0)"
                    "\nraise SystemExit(1)"
                )
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(LANDLOCK_EXEC),
                        "--allow-read",
                        "/usr",
                        "--allow-read",
                        "/etc/ld.so.cache",
                        "--allow-read",
                        allowed_directory,
                        "--",
                        sys.executable,
                        "-I",
                        "-c",
                        probe,
                        str(denied_file),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )

        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
