from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    from scripts.inspect_model import (
        DEFAULT_MODEL,
        DEFAULT_SOURCE_URL,
        sha256_argument,
        sha256_file,
        trusted_sha256_for,
    )
except ModuleNotFoundError:  # Direct execution from scripts/.
    from inspect_model import (  # type: ignore[no-redef]
        DEFAULT_MODEL,
        DEFAULT_SOURCE_URL,
        sha256_argument,
        sha256_file,
        trusted_sha256_for,
    )


ROOT = Path(__file__).resolve().parents[1]
ENTRY_SCRIPT = ROOT / "scripts" / "sandbox_entry.sh"
WORKER_SCRIPT = ROOT / "scripts" / "architecture_worker.py"
LANDLOCK_SCRIPT = ROOT / "scripts" / "landlock_exec.py"
DEFAULT_VENV = ROOT / ".venv"
DEFAULT_REPORT = ROOT / "outputs" / "reports" / "architecture_report.json"
SANDBOX_PROFILE = "linux-userns-landlock-v1"
REQUIRED_COMMANDS = ("unshare", "setpriv", "prlimit")


class SandboxConfigurationError(RuntimeError):
    """Raised when the hardened execution boundary cannot be established."""


def analysis_python_version(venv_path: Path) -> tuple[int, int, int]:
    completed = subprocess.run(
        [
            str(venv_path / "bin" / "python"),
            "-I",
            "-c",
            "import sys; print('.'.join(map(str, sys.version_info[:3])))",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "unknown error"
        raise SandboxConfigurationError(f"Analysis Python is not runnable: {detail}")
    try:
        major, minor, patch = (int(part) for part in completed.stdout.strip().split("."))
    except ValueError as exc:
        raise SandboxConfigurationError("Analysis Python returned an invalid version.") from exc
    return major, minor, patch


def validate_model_python_compatibility(
    model_filename: str,
    python_version: tuple[int, int, int],
) -> None:
    major_minor = python_version[:2]
    if model_filename == "model_3_11.pt" and major_minor != (3, 11):
        raise SandboxConfigurationError(
            "model_3_11.pt contains Python 3.11 cloudpickled bytecode and must be "
            f"analyzed with Python 3.11, not {python_version[0]}.{python_version[1]}."
        )
    if model_filename == "model.pt" and major_minor >= (3, 11):
        raise SandboxConfigurationError(
            "model.pt is the pre-3.11 artifact; use a Python version older than 3.11."
        )


def required_command_paths() -> dict[str, str]:
    missing: list[str] = []
    resolved: dict[str, str] = {}
    for command in REQUIRED_COMMANDS:
        path = shutil.which(command)
        if path is None:
            missing.append(command)
        else:
            resolved[command] = path
    if missing:
        raise SandboxConfigurationError(
            f"Missing sandbox commands: {', '.join(sorted(missing))}"
        )
    return resolved


def probe_namespace_support(unshare_path: str) -> None:
    completed = subprocess.run(
        [
            unshare_path,
            "--user",
            "--map-root-user",
            "--mount",
            "--net",
            "--pid",
            "--ipc",
            "--uts",
            "--fork",
            "--kill-child=SIGKILL",
            "--",
            "/usr/bin/true",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        raise SandboxConfigurationError(f"Linux namespace probe failed: {detail}")


def probe_landlock_support() -> int:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "landlock_exec.py"), "--probe"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        raise SandboxConfigurationError(f"Landlock probe failed: {detail}")
    try:
        abi_version = int(json.loads(completed.stdout)["landlock_abi"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SandboxConfigurationError("Landlock probe returned invalid output.") from exc
    if abi_version < 3:
        raise SandboxConfigurationError(
            f"Landlock ABI {abi_version} is too old; ABI 3 or newer is required."
        )
    return abi_version


def build_unshare_command(
    *,
    unshare_path: str,
    sandbox_root: Path,
    venv_path: Path,
    model_path: Path,
    output_path: Path,
    source_url: str,
    expected_sha256: str,
    cpu_seconds: int,
    memory_bytes: int,
    output_limit_bytes: int,
) -> list[str]:
    return [
        unshare_path,
        "--user",
        "--map-root-user",
        "--mount",
        "--net",
        "--pid",
        "--ipc",
        "--uts",
        "--fork",
        "--kill-child=SIGKILL",
        "--propagation",
        "private",
        "--",
        "/bin/sh",
        str(ENTRY_SCRIPT),
        str(sandbox_root),
        str(venv_path),
        str(model_path),
        str(ROOT / "scripts"),
        str(output_path.parent),
        output_path.name,
        model_path.name,
        source_url,
        expected_sha256,
        str(cpu_seconds),
        str(memory_bytes),
        str(output_limit_bytes),
        SANDBOX_PROFILE,
    ]


def validate_architecture_report(report: Any, expected_sha256: str) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise ValueError("Architecture report must be a JSON object.")
    if report.get("schema_version") != 1:
        raise ValueError("Unsupported architecture report schema.")
    artifact = report.get("artifact")
    if not isinstance(artifact, dict) or artifact.get("sha256") != expected_sha256:
        raise ValueError("Architecture report artifact hash does not match the trusted hash.")
    sandbox = report.get("sandbox")
    if not isinstance(sandbox, dict) or sandbox.get("profile") != SANDBOX_PROFILE:
        raise ValueError("Architecture report was not produced by the required sandbox profile.")
    if report.get("inference_executed") is not False:
        raise ValueError("Architecture report does not confirm that inference was skipped.")
    if not isinstance(report.get("loaded_object"), dict):
        raise ValueError("Architecture report is missing loaded-object metadata.")
    if not isinstance(report.get("callables"), list):
        raise ValueError("Architecture report is missing required analysis sections.")
    if not isinstance(report.get("final_predicate_circuit"), dict):
        raise ValueError("Architecture report is missing the final predicate derivation.")
    return report


def publish_report(
    staged_report: Path,
    destination: Path,
    expected_sha256: str,
    maximum_bytes: int,
) -> None:
    staged_stat = staged_report.lstat()
    if not stat.S_ISREG(staged_stat.st_mode) or staged_stat.st_nlink != 1:
        raise ValueError("Sandbox report must be a single regular file, not a link.")
    if staged_stat.st_size > maximum_bytes:
        raise ValueError("Sandbox report exceeds the configured output limit.")
    descriptor = os.open(staged_report, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
        serialized_report = handle.read(maximum_bytes + 1)
    if len(serialized_report.encode("utf-8")) > maximum_bytes:
        raise ValueError("Sandbox report exceeds the configured output limit.")
    report = validate_architecture_report(
        json.loads(serialized_report),
        expected_sha256,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(report, handle, indent=2)
        handle.write("\n")
    os.replace(temporary_path, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and inspect the Jane Street model inside an isolated Linux "
            "user/mount/PID/network namespace with a Landlock filesystem allow-list."
        )
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--expected-sha256", type=sha256_argument, default=None)
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--cpu-seconds", type=int, default=900)
    parser.add_argument("--memory-gib", type=int, default=8)
    parser.add_argument("--output-mib", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if platform.system() != "Linux":
        print("The architecture sandbox currently requires Linux.", file=sys.stderr)
        return 2

    model_path = args.model.resolve()
    venv_path = args.venv.resolve()
    report_path = args.report.resolve()
    expected_sha256 = trusted_sha256_for(model_path, args.expected_sha256)

    if not model_path.is_file():
        print(f"Model artifact not found: {model_path}", file=sys.stderr)
        return 2
    if expected_sha256 is None:
        print(
            "No trusted digest is registered for this artifact; provide --expected-sha256.",
            file=sys.stderr,
        )
        return 2
    if not (venv_path / "bin" / "python").exists():
        print(
            f"Analysis environment not found at {venv_path}. Install requirements/analysis-cpu.txt first.",
            file=sys.stderr,
        )
        return 2
    if not all(path.is_file() for path in (ENTRY_SCRIPT, WORKER_SCRIPT, LANDLOCK_SCRIPT)):
        print("Sandbox implementation files are missing.", file=sys.stderr)
        return 2
    if min(args.cpu_seconds, args.memory_gib, args.output_mib) <= 0:
        print("Resource limits must be positive integers.", file=sys.stderr)
        return 2

    try:
        commands = required_command_paths()
        probe_namespace_support(commands["unshare"])
        landlock_abi = probe_landlock_support()
        python_version = analysis_python_version(venv_path)
        validate_model_python_compatibility(model_path.name, python_version)
    except SandboxConfigurationError as exc:
        print(f"Sandbox unavailable: {exc}", file=sys.stderr)
        return 2

    print(f"Hashing {model_path.name} before entering the sandbox...")
    print(
        f"Using analysis Python {python_version[0]}.{python_version[1]}.{python_version[2]}."
    )
    actual_sha256 = sha256_file(model_path)
    if actual_sha256 != expected_sha256:
        print(
            f"Trusted hash mismatch: expected {expected_sha256}, got {actual_sha256}",
            file=sys.stderr,
        )
        return 2

    with tempfile.TemporaryDirectory(prefix="jsmi-architecture-", dir="/tmp") as directory:
        staging_root = Path(directory)
        sandbox_root = staging_root / "private-tmp"
        sandbox_output = staging_root / "output"
        sandbox_input = staging_root / "input"
        sandbox_root.mkdir(mode=0o700)
        sandbox_output.mkdir(mode=0o700)
        sandbox_input.mkdir(mode=0o700)
        staged_report = sandbox_output / report_path.name
        staged_model = sandbox_input / model_path.name
        print("Copying the verified artifact into the private sandbox staging area...")
        shutil.copyfile(model_path, staged_model)
        staged_model.chmod(0o400)

        command = build_unshare_command(
            unshare_path=commands["unshare"],
            sandbox_root=sandbox_root,
            venv_path=venv_path,
            model_path=staged_model,
            output_path=staged_report,
            source_url=args.source_url,
            expected_sha256=expected_sha256,
            cpu_seconds=args.cpu_seconds,
            memory_bytes=args.memory_gib * 1024**3,
            output_limit_bytes=args.output_mib * 1024**2,
        )
        environment = {
            "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        }
        print(
            f"Starting isolated architecture worker "
            f"(Landlock ABI {landlock_abi}; network namespace has no interfaces)..."
        )
        completed = subprocess.run(command, env=environment, check=False)
        if completed.returncode != 0:
            print(
                f"Sandboxed architecture worker failed with exit status {completed.returncode}.",
                file=sys.stderr,
            )
            return completed.returncode or 1
        if not staged_report.is_file():
            print("Sandboxed worker did not produce a report.", file=sys.stderr)
            return 1
        try:
            publish_report(
                staged_report,
                report_path,
                expected_sha256,
                args.output_mib * 1024**2,
            )
        except (OSError, ValueError, UnicodeError, RecursionError, json.JSONDecodeError) as exc:
            print(f"Rejected sandbox report: {exc}", file=sys.stderr)
            return 1

    print(f"Verified architecture report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
