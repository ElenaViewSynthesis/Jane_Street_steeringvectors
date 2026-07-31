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
    from scripts.activation_schema import (
        SANDBOX_PROFILE,
        validate_activation_report,
    )
    from scripts.inspect_model import (
        DEFAULT_MODEL,
        DEFAULT_SOURCE_URL,
        sha256_argument,
        sha256_file,
        trusted_sha256_for,
    )
    from scripts.probe_schema import (
        ProbeSchemaError,
        load_probe_manifest,
        loads_json,
    )
    from scripts.run_architecture_sandbox import (
        SandboxConfigurationError,
        analysis_python_version,
        probe_landlock_support,
        probe_namespace_support,
        required_command_paths,
        validate_model_python_compatibility,
    )
except ModuleNotFoundError:  # Direct execution from scripts/.
    from activation_schema import (  # type: ignore[no-redef]
        SANDBOX_PROFILE,
        validate_activation_report,
    )
    from inspect_model import (  # type: ignore[no-redef]
        DEFAULT_MODEL,
        DEFAULT_SOURCE_URL,
        sha256_argument,
        sha256_file,
        trusted_sha256_for,
    )
    from probe_schema import (  # type: ignore[no-redef]
        ProbeSchemaError,
        load_probe_manifest,
        loads_json,
    )
    from run_architecture_sandbox import (  # type: ignore[no-redef]
        SandboxConfigurationError,
        analysis_python_version,
        probe_landlock_support,
        probe_namespace_support,
        required_command_paths,
        validate_model_python_compatibility,
    )


ROOT = Path(__file__).resolve().parents[1]
ENTRY_SCRIPT = ROOT / "scripts" / "activation_sandbox_entry.sh"
WORKER_SCRIPT = ROOT / "scripts" / "activation_worker.py"
LANDLOCK_SCRIPT = ROOT / "scripts" / "landlock_exec.py"
ACTIVATION_SCHEMA_SCRIPT = ROOT / "scripts" / "activation_schema.py"
PROBE_SCHEMA_SCRIPT = ROOT / "scripts" / "probe_schema.py"
DEFAULT_MANIFEST = (
    ROOT / "experiments" / "activations" / "m4-capture-smoke-v1.json"
)
DEFAULT_REPORT_DIRECTORY = ROOT / "outputs" / "activations"
MAX_REPETITIONS = 10


def build_activation_unshare_command(
    *,
    unshare_path: str,
    sandbox_root: Path,
    venv_path: Path,
    model_path: Path,
    manifest_path: Path,
    output_path: Path,
    source_url: str,
    expected_sha256: str,
    expected_manifest_sha256: str,
    repetitions: int,
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
        str(manifest_path),
        str(ROOT / "scripts"),
        str(output_path.parent),
        output_path.name,
        model_path.name,
        source_url,
        expected_sha256,
        expected_manifest_sha256,
        str(repetitions),
        str(cpu_seconds),
        str(memory_bytes),
        str(output_limit_bytes),
        SANDBOX_PROFILE,
    ]


def publish_activation_report(
    staged_report: Path,
    destination: Path,
    *,
    maximum_bytes: int,
    expected_sha256: str,
    expected_manifest_sha256: str,
    manifest: dict[str, Any],
    repetitions: int,
) -> dict[str, Any]:
    metadata = staged_report.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ProbeSchemaError("Sandbox report must be a single regular file, not a link.")
    if metadata.st_size > maximum_bytes:
        raise ProbeSchemaError("Sandbox report exceeds the configured output limit.")
    descriptor = os.open(staged_report, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as handle:
        payload = handle.read(maximum_bytes + 1)
    if len(payload) > maximum_bytes:
        raise ProbeSchemaError("Sandbox report exceeds the configured output limit.")
    report = validate_activation_report(
        loads_json(payload),
        expected_sha256=expected_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        manifest=manifest,
        repetitions=repetitions,
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
        json.dump(report, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary_path, destination)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture the final hidden representation and predicate activations inside "
            "isolated Linux namespaces and a Landlock filesystem allow-list."
        )
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--expected-sha256", type=sha256_argument, default=None)
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--cpu-seconds", type=int, default=900)
    parser.add_argument("--memory-gib", type=int, default=8)
    parser.add_argument("--output-mib", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if platform.system() != "Linux":
        print("The activation sandbox currently requires Linux.", file=sys.stderr)
        return 2

    model_path = args.model.resolve()
    venv_path = args.venv.resolve()
    manifest_path = args.manifest.resolve()
    expected_sha256 = trusted_sha256_for(model_path, args.expected_sha256)
    try:
        manifest, manifest_payload, manifest_sha256 = load_probe_manifest(manifest_path)
    except ProbeSchemaError as exc:
        print(f"Activation manifest rejected: {exc}", file=sys.stderr)
        return 2
    report_path = (
        args.report.resolve()
        if args.report is not None
        else DEFAULT_REPORT_DIRECTORY / f"{manifest['suite_id']}-activations-v1.json"
    )

    if not model_path.is_file():
        print(f"Model artifact not found: {model_path}", file=sys.stderr)
        return 2
    if expected_sha256 is None:
        print("No trusted model digest is available.", file=sys.stderr)
        return 2
    if report_path in {model_path, manifest_path}:
        print("Activation report cannot overwrite an input file.", file=sys.stderr)
        return 2
    if not (venv_path / "bin" / "python").exists():
        print(f"Analysis environment not found at {venv_path}.", file=sys.stderr)
        return 2
    required_files = (
        ENTRY_SCRIPT,
        WORKER_SCRIPT,
        LANDLOCK_SCRIPT,
        ACTIVATION_SCHEMA_SCRIPT,
        PROBE_SCHEMA_SCRIPT,
    )
    if not all(path.is_file() for path in required_files):
        print("Activation sandbox implementation files are missing.", file=sys.stderr)
        return 2
    if not 2 <= args.repetitions <= MAX_REPETITIONS:
        print(f"Repetitions must be between 2 and {MAX_REPETITIONS}.", file=sys.stderr)
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

    print(f"Hashing {model_path.name} before entering the activation sandbox...")
    actual_sha256 = sha256_file(model_path)
    if actual_sha256 != expected_sha256:
        print(
            f"Trusted hash mismatch: expected {expected_sha256}, got {actual_sha256}",
            file=sys.stderr,
        )
        return 2

    with tempfile.TemporaryDirectory(prefix="jsmi-activation-", dir="/tmp") as directory:
        staging_root = Path(directory)
        sandbox_root = staging_root / "private-tmp"
        sandbox_output = staging_root / "output"
        sandbox_input = staging_root / "input"
        sandbox_root.mkdir(mode=0o700)
        sandbox_output.mkdir(mode=0o700)
        sandbox_input.mkdir(mode=0o700)
        staged_report = sandbox_output / report_path.name
        staged_model = sandbox_input / model_path.name
        staged_manifest = sandbox_input / manifest_path.name

        print("Copying verified activation inputs into private sandbox staging...")
        shutil.copyfile(model_path, staged_model)
        staged_model.chmod(0o400)
        staged_manifest.write_bytes(manifest_payload)
        staged_manifest.chmod(0o400)

        command = build_activation_unshare_command(
            unshare_path=commands["unshare"],
            sandbox_root=sandbox_root,
            venv_path=venv_path,
            model_path=staged_model,
            manifest_path=staged_manifest,
            output_path=staged_report,
            source_url=args.source_url,
            expected_sha256=expected_sha256,
            expected_manifest_sha256=manifest_sha256,
            repetitions=args.repetitions,
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
            f"Capturing {len(manifest['cases'])} cases x {args.repetitions} repetitions "
            f"(Landlock ABI {landlock_abi}; no network interfaces)..."
        )
        completed = subprocess.run(command, env=environment, check=False)
        if completed.returncode != 0:
            print(
                f"Sandboxed activation worker failed with exit status "
                f"{completed.returncode}.",
                file=sys.stderr,
            )
            return completed.returncode or 1
        if not staged_report.is_file():
            print("Sandboxed activation worker did not produce a report.", file=sys.stderr)
            return 1
        try:
            report = publish_activation_report(
                staged_report,
                report_path,
                maximum_bytes=args.output_mib * 1024**2,
                expected_sha256=expected_sha256,
                expected_manifest_sha256=manifest_sha256,
                manifest=manifest,
                repetitions=args.repetitions,
            )
        except (OSError, ProbeSchemaError, RecursionError) as exc:
            print(f"Rejected sandbox report: {exc}", file=sys.stderr)
            return 1

    summary = report["summary"]
    print(f"Verified activation report: {report_path}")
    print(
        f"Captured {summary['observation_count']} observations; all deterministic: "
        f"{summary['all_cases_deterministic']}; predicate-match histogram: "
        f"{summary['matched_predicate_count_histogram']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
