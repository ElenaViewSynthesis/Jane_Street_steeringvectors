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
    from scripts.analyze_activation_report import load_validated_activation_report
    from scripts.inspect_model import (
        DEFAULT_MODEL,
        DEFAULT_SOURCE_URL,
        sha256_argument,
        sha256_file,
        trusted_sha256_for,
    )
    from scripts.intervention_schema import (
        MAX_INTERVENTIONS,
        validate_intervention_report,
        load_intervention_spec,
    )
    from scripts.probe_schema import (
        ProbeSchemaError,
        load_probe_manifest,
        loads_json,
        read_bounded_regular_file,
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
    from analyze_activation_report import load_validated_activation_report  # type: ignore[no-redef]
    from inspect_model import (  # type: ignore[no-redef]
        DEFAULT_MODEL,
        DEFAULT_SOURCE_URL,
        sha256_argument,
        sha256_file,
        trusted_sha256_for,
    )
    from intervention_schema import (  # type: ignore[no-redef]
        MAX_INTERVENTIONS,
        validate_intervention_report,
        load_intervention_spec,
    )
    from probe_schema import (  # type: ignore[no-redef]
        ProbeSchemaError,
        load_probe_manifest,
        loads_json,
        read_bounded_regular_file,
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
ENTRY_SCRIPT = ROOT / "scripts" / "intervention_sandbox_entry.sh"
WORKER_SCRIPT = ROOT / "scripts" / "intervention_worker.py"
LANDLOCK_SCRIPT = ROOT / "scripts" / "landlock_exec.py"
ACTIVATION_SCHEMA_SCRIPT = ROOT / "scripts" / "activation_schema.py"
INTERVENTION_SCHEMA_SCRIPT = ROOT / "scripts" / "intervention_schema.py"
PROBE_SCHEMA_SCRIPT = ROOT / "scripts" / "probe_schema.py"
DEFAULT_REPORT_DIRECTORY = ROOT / "outputs" / "interventions"
MAX_SPEC_BYTES = 4 * 1024 * 1024
MAX_REPETITIONS = 10
SANDBOX_PROFILE = "linux-userns-landlock-intervention-v1"


def build_intervention_unshare_command(
    *,
    unshare_path: str,
    sandbox_root: Path,
    venv_path: Path,
    model_path: Path,
    manifest_path: Path,
    spec_path: Path,
    output_path: Path,
    source_url: str,
    expected_sha256: str,
    expected_manifest_sha256: str,
    expected_spec_sha256: str,
    repetitions: int,
    cpu_seconds: int,
    memory_bytes: int,
    output_limit_bytes: int,
) -> list[str]:
    return [
        unshare_path, "--user", "--map-root-user", "--mount", "--net", "--pid",
        "--ipc", "--uts", "--fork", "--kill-child=SIGKILL", "--propagation", "private",
        "--", "/bin/sh", str(ENTRY_SCRIPT), str(sandbox_root), str(venv_path),
        str(model_path), str(manifest_path), str(spec_path), str(ROOT / "scripts"),
        str(output_path.parent), output_path.name, model_path.name, source_url,
        expected_sha256, expected_manifest_sha256, expected_spec_sha256, str(repetitions),
        str(cpu_seconds), str(memory_bytes), str(output_limit_bytes), SANDBOX_PROFILE,
    ]


def publish_intervention_report(
    staged_report: Path,
    destination: Path,
    *,
    maximum_bytes: int,
    manifest: dict[str, Any],
    expected_model_sha256: str,
    expected_spec_sha256: str,
    spec: dict[str, Any],
    expected_source_report_sha256: str,
    repetitions: int,
) -> dict[str, Any]:
    metadata = staged_report.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ProbeSchemaError("Sandbox intervention report must be a single regular file, not a link.")
    if metadata.st_size > maximum_bytes:
        raise ProbeSchemaError("Sandbox intervention report exceeds the configured output limit.")
    descriptor = os.open(staged_report, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as handle:
        payload = handle.read(maximum_bytes + 1)
    if len(payload) > maximum_bytes:
        raise ProbeSchemaError("Sandbox intervention report exceeds the configured output limit.")
    report = validate_intervention_report(
        loads_json(payload), manifest=manifest, expected_model_sha256=expected_model_sha256,
        expected_spec_sha256=expected_spec_sha256, spec=spec,
        expected_source_report_sha256=expected_source_report_sha256, repetitions=repetitions,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent,
                                     prefix=f".{destination.name}.", suffix=".tmp", delete=False) as handle:
        temporary_path = Path(handle.name)
        json.dump(report, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary_path, destination)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hash-bound h192 interventions in Linux namespaces and Landlock."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-activation-report", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
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
        print("The intervention sandbox currently requires Linux.", file=sys.stderr)
        return 2
    model_path, venv_path = args.model.resolve(), args.venv.resolve()
    manifest_path, source_path, spec_path = (
        args.manifest.resolve(), args.source_activation_report.resolve(), args.spec.resolve()
    )
    expected_model_sha256 = trusted_sha256_for(model_path, args.expected_sha256)
    try:
        manifest, manifest_payload, manifest_sha256 = load_probe_manifest(manifest_path)
        _source_report, source_report_sha256 = load_validated_activation_report(source_path, manifest_path)
        spec_payload = read_bounded_regular_file(spec_path, MAX_SPEC_BYTES)
        spec, spec_sha256 = load_intervention_spec(
            spec_payload, case_ids={case["id"] for case in manifest["cases"]}
        )
    except (OSError, ProbeSchemaError, RecursionError) as exc:
        print(f"Intervention input rejected: {exc}", file=sys.stderr)
        return 2
    report_path = (args.report.resolve() if args.report else
                   DEFAULT_REPORT_DIRECTORY / f"{spec['suite_id']}-interventions-v1.json")
    if spec["source_manifest_sha256"] != manifest_sha256:
        print("Intervention spec is not bound to the supplied manifest.", file=sys.stderr)
        return 2
    if spec["source_activation_report_sha256"] != source_report_sha256:
        print("Intervention spec is not bound to the supplied activation report.", file=sys.stderr)
        return 2
    if expected_model_sha256 is None or not model_path.is_file():
        print("Model artifact or trusted hash is unavailable.", file=sys.stderr)
        return 2
    if report_path in {model_path, manifest_path, source_path, spec_path}:
        print("Intervention report cannot overwrite an input file.", file=sys.stderr)
        return 2
    if not (venv_path / "bin" / "python").exists():
        print(f"Analysis environment not found at {venv_path}.", file=sys.stderr)
        return 2
    required = (ENTRY_SCRIPT, WORKER_SCRIPT, LANDLOCK_SCRIPT, ACTIVATION_SCHEMA_SCRIPT,
                INTERVENTION_SCHEMA_SCRIPT, PROBE_SCHEMA_SCRIPT)
    if not all(path.is_file() for path in required):
        print("Intervention sandbox implementation files are missing.", file=sys.stderr)
        return 2
    if not 2 <= args.repetitions <= MAX_REPETITIONS or min(args.cpu_seconds, args.memory_gib, args.output_mib) <= 0:
        print("Intervention repetitions and resource limits are invalid.", file=sys.stderr)
        return 2
    try:
        commands = required_command_paths()
        probe_namespace_support(commands["unshare"])
        landlock_abi = probe_landlock_support()
        validate_model_python_compatibility(model_path.name, analysis_python_version(venv_path))
    except SandboxConfigurationError as exc:
        print(f"Sandbox unavailable: {exc}", file=sys.stderr)
        return 2
    print(f"Hashing {model_path.name} before entering the intervention sandbox...")
    if sha256_file(model_path) != expected_model_sha256:
        print("Trusted model hash mismatch.", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory(prefix="jsmi-intervention-", dir="/tmp") as directory:
        staging_root = Path(directory)
        sandbox_root, sandbox_output, sandbox_input = (
            staging_root / "private-tmp", staging_root / "output", staging_root / "input"
        )
        for path in (sandbox_root, sandbox_output, sandbox_input):
            path.mkdir(mode=0o700)
        staged_report = sandbox_output / report_path.name
        staged_model, staged_manifest, staged_spec = (
            sandbox_input / model_path.name, sandbox_input / manifest_path.name, sandbox_input / spec_path.name
        )
        print("Copying verified intervention inputs into private sandbox staging...")
        shutil.copyfile(model_path, staged_model)
        staged_model.chmod(0o400)
        staged_manifest.write_bytes(manifest_payload)
        staged_manifest.chmod(0o400)
        staged_spec.write_bytes(spec_payload)
        staged_spec.chmod(0o400)
        command = build_intervention_unshare_command(
            unshare_path=commands["unshare"], sandbox_root=sandbox_root, venv_path=venv_path,
            model_path=staged_model, manifest_path=staged_manifest, spec_path=staged_spec,
            output_path=staged_report, source_url=args.source_url, expected_sha256=expected_model_sha256,
            expected_manifest_sha256=manifest_sha256, expected_spec_sha256=spec_sha256,
            repetitions=args.repetitions, cpu_seconds=args.cpu_seconds,
            memory_bytes=args.memory_gib * 1024**3, output_limit_bytes=args.output_mib * 1024**2,
        )
        environment = {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
        print(f"Intervening on {len(manifest['cases'])} cases (Landlock ABI {landlock_abi}; no network interfaces)...")
        completed = subprocess.run(command, env=environment, check=False)
        if completed.returncode != 0:
            print(f"Sandboxed intervention worker failed with exit status {completed.returncode}.", file=sys.stderr)
            return completed.returncode or 1
        if not staged_report.is_file():
            print("Sandboxed intervention worker did not produce a report.", file=sys.stderr)
            return 1
        try:
            report = publish_intervention_report(
                staged_report, report_path, maximum_bytes=args.output_mib * 1024**2,
                manifest=manifest, expected_model_sha256=expected_model_sha256,
                expected_spec_sha256=spec_sha256, spec=spec,
                expected_source_report_sha256=source_report_sha256, repetitions=args.repetitions,
            )
        except (OSError, ProbeSchemaError, RecursionError) as exc:
            print(f"Rejected sandbox intervention report: {exc}", file=sys.stderr)
            return 1
    print(f"Verified intervention report: {report_path}")
    print(f"Captured {report['summary']['intervention_observation_count']} intervention observations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
