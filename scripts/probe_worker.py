from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

try:
    from scripts.inspect_model import sha256_argument
    from scripts.probe_schema import (
        REPORT_SCHEMA_VERSION,
        ProbeSchemaError,
        load_probe_manifest,
        summarize_probe_results,
    )
except ModuleNotFoundError:  # Direct execution from scripts/.
    from inspect_model import sha256_argument  # type: ignore[no-redef]
    from probe_schema import (  # type: ignore[no-redef]
        REPORT_SCHEMA_VERSION,
        ProbeSchemaError,
        load_probe_manifest,
        summarize_probe_results,
    )


SANDBOX_PROFILE = "linux-userns-landlock-probe-v1"


class ProbeExecutionError(RuntimeError):
    """Raised when the model does not satisfy the scalar probe contract."""


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_observation(output: Any, torch: Any, repetition: int) -> dict[str, Any]:
    if not isinstance(output, torch.Tensor):
        raise ProbeExecutionError(
            f"Model returned {type(output).__module__}.{type(output).__qualname__}, "
            "not a torch.Tensor."
        )
    if output.numel() != 1:
        raise ProbeExecutionError(
            f"Model output must contain one scalar value, not {output.numel()}."
        )
    detached = output.detach().cpu()
    value = float(detached.reshape(-1)[0].item())
    if not math.isfinite(value):
        raise ProbeExecutionError("Model returned a non-finite scalar value.")
    return {
        "repetition": repetition,
        "value": value,
        "float_hex": value.hex(),
        "dtype": str(output.dtype),
        "shape": list(output.shape),
        "device": str(output.device),
    }


def run_probe_cases(
    model: Any,
    torch: Any,
    manifest: dict[str, Any],
    repetitions: int,
) -> list[dict[str, Any]]:
    results = [
        {
            "case": case,
            "input_length": len(case["input"]),
            "observations": [],
            "deterministic": False,
        }
        for case in manifest["cases"]
    ]

    with torch.inference_mode():
        if torch.is_grad_enabled():
            raise ProbeExecutionError("Gradient tracking remained enabled in inference mode.")
        for repetition in range(repetitions):
            for result in results:
                try:
                    output = model(result["case"]["input"])
                    observation = scalar_observation(output, torch, repetition)
                except Exception as exc:
                    case_id = result["case"]["id"]
                    raise ProbeExecutionError(
                        f"Probe case {case_id!r} failed during repetition {repetition}."
                    ) from exc
                result["observations"].append(observation)

    for result in results:
        signatures = {
            (
                observation["float_hex"],
                observation["dtype"],
                tuple(observation["shape"]),
                observation["device"],
            )
            for observation in result["observations"]
        }
        result["deterministic"] = len(signatures) == 1
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--expected-sha256", type=sha256_argument, required=True)
    parser.add_argument("--expected-manifest-sha256", type=sha256_argument, required=True)
    parser.add_argument("--repetitions", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sandbox_profile = os.environ.get("JSMI_SANDBOX_PROFILE")
    if sandbox_profile != SANDBOX_PROFILE:
        print("Probe worker refuses to run outside the required sandbox.", file=sys.stderr)
        return 2
    if not 2 <= args.repetitions <= 10:
        print("Probe repetitions must be between 2 and 10.", file=sys.stderr)
        return 2

    actual_sha256 = sha256_file(args.model)
    if actual_sha256 != args.expected_sha256:
        print("Artifact hash changed inside the sandbox; refusing to unpickle.", file=sys.stderr)
        return 2
    try:
        manifest, _manifest_payload, manifest_sha256 = load_probe_manifest(args.manifest)
    except ProbeSchemaError as exc:
        print(f"Probe manifest rejected inside sandbox: {exc}", file=sys.stderr)
        return 2
    if manifest_sha256 != args.expected_manifest_sha256:
        print("Probe manifest hash changed inside the sandbox.", file=sys.stderr)
        return 2

    import cloudpickle
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    model = torch.load(args.model, map_location="cpu", weights_only=False)
    if not isinstance(model, torch.nn.Module):
        print("Loaded object is not a torch.nn.Module.", file=sys.stderr)
        return 2
    model.eval()

    try:
        results = run_probe_cases(model, torch, manifest, args.repetitions)
    except ProbeExecutionError as exc:
        print(f"Probe execution failed: {exc}", file=sys.stderr)
        return 1

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact": {
            "filename": args.artifact_name,
            "size_bytes": args.model.stat().st_size,
            "sha256": actual_sha256,
            "source_url": args.source_url,
        },
        "probe_suite": {
            "suite_id": manifest["suite_id"],
            "description": manifest["description"],
            "manifest_filename": args.manifest.name,
            "manifest_sha256": manifest_sha256,
            "case_count": len(manifest["cases"]),
            "repetitions": args.repetitions,
        },
        "sandbox": {
            "profile": sandbox_profile,
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
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cloudpickle": cloudpickle.__version__,
            "platform": platform.platform(),
        },
        "execution": {
            "inference_mode": True,
            "model_eval": model.training is False,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "seed": 0,
        },
        "results": results,
        "summary": summarize_probe_results(results),
        "inference_executed": True,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(f"{args.output.suffix}.tmp")
    temporary_output.write_text(
        json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_output, args.output)
    print(f"Probe report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
