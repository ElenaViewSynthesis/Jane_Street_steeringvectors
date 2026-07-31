from __future__ import annotations

import argparse
import json
import math
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
    from scripts.probe_schema import (
        REPORT_SCHEMA_VERSION,
        ProbeSchemaError,
        load_probe_manifest,
        loads_json,
        summarize_probe_results,
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
    from inspect_model import (  # type: ignore[no-redef]
        DEFAULT_MODEL,
        DEFAULT_SOURCE_URL,
        sha256_argument,
        sha256_file,
        trusted_sha256_for,
    )
    from probe_schema import (  # type: ignore[no-redef]
        REPORT_SCHEMA_VERSION,
        ProbeSchemaError,
        load_probe_manifest,
        loads_json,
        summarize_probe_results,
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
ENTRY_SCRIPT = ROOT / "scripts" / "probe_sandbox_entry.sh"
WORKER_SCRIPT = ROOT / "scripts" / "probe_worker.py"
LANDLOCK_SCRIPT = ROOT / "scripts" / "landlock_exec.py"
SCHEMA_SCRIPT = ROOT / "scripts" / "probe_schema.py"
DEFAULT_VENV = ROOT / ".venv"
DEFAULT_MANIFEST = ROOT / "experiments" / "probes" / "m3-smoke-v1.json"
DEFAULT_REPORT_DIRECTORY = ROOT / "outputs" / "probes"
SANDBOX_PROFILE = "linux-userns-landlock-probe-v1"
MAX_REPETITIONS = 10


def build_probe_unshare_command(
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


def _validate_observation(observation: Any, repetition: int) -> None:
    if not isinstance(observation, dict):
        raise ProbeSchemaError("Probe observation must be a JSON object.")
    required = {"repetition", "value", "float_hex", "dtype", "shape", "device"}
    if set(observation) != required:
        raise ProbeSchemaError("Probe observation has an invalid field set.")
    if observation["repetition"] != repetition:
        raise ProbeSchemaError("Probe observation repetition is out of order.")
    value = observation["value"]
    if type(value) not in {int, float} or not math.isfinite(value):
        raise ProbeSchemaError("Probe observation value must be finite and numeric.")
    value = float(value)
    if observation["float_hex"] != value.hex():
        raise ProbeSchemaError("Probe observation float encoding is inconsistent.")
    if not isinstance(observation["dtype"], str) or not observation["dtype"]:
        raise ProbeSchemaError("Probe observation dtype is invalid.")
    if not isinstance(observation["device"], str) or not observation["device"]:
        raise ProbeSchemaError("Probe observation device is invalid.")
    shape = observation["shape"]
    if not isinstance(shape, list) or any(
        type(dimension) is not int or dimension < 0 for dimension in shape
    ):
        raise ProbeSchemaError("Probe observation shape is invalid.")
    element_count = 1
    for dimension in shape:
        element_count *= dimension
    if element_count != 1:
        raise ProbeSchemaError("Probe observation does not describe a scalar tensor.")


def validate_probe_report(
    report: Any,
    *,
    expected_sha256: str,
    expected_manifest_sha256: str,
    manifest: dict[str, Any],
    repetitions: int,
) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise ProbeSchemaError("Probe report must be a JSON object.")
    required_keys = {
        "schema_version",
        "artifact",
        "probe_suite",
        "sandbox",
        "runtime",
        "execution",
        "results",
        "summary",
        "inference_executed",
    }
    if set(report) != required_keys:
        raise ProbeSchemaError("Probe report has an invalid top-level field set.")
    if (
        type(report["schema_version"]) is not int
        or report["schema_version"] != REPORT_SCHEMA_VERSION
    ):
        raise ProbeSchemaError("Unsupported probe report schema version.")
    artifact = report["artifact"]
    if not isinstance(artifact, dict) or set(artifact) != {
        "filename",
        "size_bytes",
        "sha256",
        "source_url",
    }:
        raise ProbeSchemaError("Probe report artifact metadata is invalid.")
    if artifact.get("sha256") != expected_sha256:
        raise ProbeSchemaError("Probe report artifact hash does not match.")
    suite = report["probe_suite"]
    if not isinstance(suite, dict) or set(suite) != {
        "suite_id",
        "description",
        "manifest_filename",
        "manifest_sha256",
        "case_count",
        "repetitions",
    }:
        raise ProbeSchemaError("Probe report suite metadata is missing.")
    expected_suite_fields = {
        "suite_id": manifest["suite_id"],
        "description": manifest["description"],
        "manifest_sha256": expected_manifest_sha256,
        "case_count": len(manifest["cases"]),
        "repetitions": repetitions,
    }
    for key, expected in expected_suite_fields.items():
        if suite.get(key) != expected:
            raise ProbeSchemaError(f"Probe report suite field {key!r} does not match.")
    sandbox = report["sandbox"]
    expected_sandbox = {
        "profile": SANDBOX_PROFILE,
        "network_namespace": "isolated",
        "filesystem_policy": "Landlock allow-list",
        "model_access": "read-only",
        "manifest_access": "read-only",
        "host_credentials": "not allow-listed",
        "output_access": "staging-directory-only",
        "capabilities": "dropped",
        "no_new_privileges": True,
    }
    if sandbox != expected_sandbox:
        raise ProbeSchemaError("Probe report sandbox metadata does not match.")
    runtime = report["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != {
        "python",
        "torch",
        "cloudpickle",
        "platform",
    }:
        raise ProbeSchemaError("Probe report runtime metadata is invalid.")
    if not all(isinstance(value, str) and value for value in runtime.values()):
        raise ProbeSchemaError("Probe report runtime values are invalid.")
    if report["inference_executed"] is not True:
        raise ProbeSchemaError("Probe report does not confirm inference execution.")
    execution = report["execution"]
    if not isinstance(execution, dict):
        raise ProbeSchemaError("Probe report execution metadata is missing.")
    expected_execution = {
        "inference_mode": True,
        "model_eval": True,
        "deterministic_algorithms": True,
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
        "seed": 0,
    }
    for key, expected in expected_execution.items():
        if execution.get(key) != expected:
            raise ProbeSchemaError(f"Probe execution field {key!r} does not match.")
    if set(execution) != set(expected_execution):
        raise ProbeSchemaError("Probe execution metadata has an invalid field set.")

    results = report["results"]
    if not isinstance(results, list) or len(results) != len(manifest["cases"]):
        raise ProbeSchemaError("Probe result count does not match the manifest.")
    for case, result in zip(manifest["cases"], results):
        if not isinstance(result, dict) or set(result) != {
            "case",
            "input_length",
            "observations",
            "deterministic",
        }:
            raise ProbeSchemaError("Probe result has an invalid field set.")
        if result["case"] != case or result["input_length"] != len(case["input"]):
            raise ProbeSchemaError("Probe result does not match its manifest case.")
        observations = result["observations"]
        if not isinstance(observations, list) or len(observations) != repetitions:
            raise ProbeSchemaError("Probe result repetition count does not match.")
        for repetition, observation in enumerate(observations):
            _validate_observation(observation, repetition)
        signatures = {
            (
                observation["float_hex"],
                observation["dtype"],
                tuple(observation["shape"]),
                observation["device"],
            )
            for observation in observations
        }
        if result["deterministic"] is not (len(signatures) == 1):
            raise ProbeSchemaError("Probe determinism flag is inconsistent.")

    expected_summary = summarize_probe_results(results)
    if report["summary"] != expected_summary:
        raise ProbeSchemaError("Probe summary does not match the validated observations.")
    return report


def publish_probe_report(
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
    report = validate_probe_report(
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
            "Run a deterministic text probe suite inside isolated Linux namespaces "
            "and a Landlock filesystem allow-list."
        )
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--expected-sha256", type=sha256_argument, default=None)
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--cpu-seconds", type=int, default=900)
    parser.add_argument("--memory-gib", type=int, default=8)
    parser.add_argument("--output-mib", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if platform.system() != "Linux":
        print("The probe sandbox currently requires Linux.", file=sys.stderr)
        return 2

    model_path = args.model.resolve()
    venv_path = args.venv.resolve()
    manifest_path = args.manifest.resolve()
    expected_sha256 = trusted_sha256_for(model_path, args.expected_sha256)
    try:
        manifest, manifest_payload, manifest_sha256 = load_probe_manifest(manifest_path)
    except ProbeSchemaError as exc:
        print(f"Probe manifest rejected: {exc}", file=sys.stderr)
        return 2
    report_path = (
        args.report.resolve()
        if args.report is not None
        else DEFAULT_REPORT_DIRECTORY / f"{manifest['suite_id']}.json"
    )

    if not model_path.is_file():
        print(f"Model artifact not found: {model_path}", file=sys.stderr)
        return 2
    if expected_sha256 is None:
        print("No trusted model digest is available.", file=sys.stderr)
        return 2
    if report_path in {model_path, manifest_path}:
        print("Probe report cannot overwrite an input file.", file=sys.stderr)
        return 2
    if not (venv_path / "bin" / "python").exists():
        print(f"Analysis environment not found at {venv_path}.", file=sys.stderr)
        return 2
    if not all(
        path.is_file()
        for path in (ENTRY_SCRIPT, WORKER_SCRIPT, LANDLOCK_SCRIPT, SCHEMA_SCRIPT)
    ):
        print("Probe sandbox implementation files are missing.", file=sys.stderr)
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

    print(f"Hashing {model_path.name} before entering the probe sandbox...")
    actual_sha256 = sha256_file(model_path)
    if actual_sha256 != expected_sha256:
        print(
            f"Trusted hash mismatch: expected {expected_sha256}, got {actual_sha256}",
            file=sys.stderr,
        )
        return 2

    with tempfile.TemporaryDirectory(prefix="jsmi-probe-", dir="/tmp") as directory:
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

        print("Copying verified probe inputs into private sandbox staging...")
        shutil.copyfile(model_path, staged_model)
        staged_model.chmod(0o400)
        staged_manifest.write_bytes(manifest_payload)
        staged_manifest.chmod(0o400)

        command = build_probe_unshare_command(
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
            f"Running {len(manifest['cases'])} cases x {args.repetitions} repetitions "
            f"(Landlock ABI {landlock_abi}; no network interfaces)..."
        )
        completed = subprocess.run(command, env=environment, check=False)
        if completed.returncode != 0:
            print(
                f"Sandboxed probe worker failed with exit status {completed.returncode}.",
                file=sys.stderr,
            )
            return completed.returncode or 1
        if not staged_report.is_file():
            print("Sandboxed probe worker did not produce a report.", file=sys.stderr)
            return 1
        try:
            report = publish_probe_report(
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
    print(f"Verified probe report: {report_path}")
    print(
        f"Observed {summary['unique_output_count']} unique scalar outputs across "
        f"{summary['observation_count']} observations; "
        f"all deterministic: {summary['all_cases_deterministic']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
