from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

try:
    from scripts.activation_worker import (
        ActivationExecutionError,
        ScopedActivationHooks,
        activation_observation,
        resolve_final_circuit_modules,
    )
    from scripts.inspect_model import sha256_argument
    from scripts.intervention_schema import (
        REPORT_SCHEMA_VERSION,
        derive_intervention_comparison,
        load_intervention_spec,
    )
    from scripts.probe_schema import ProbeSchemaError, load_probe_manifest, read_bounded_regular_file
except ModuleNotFoundError:  # Direct execution from scripts/.
    from activation_worker import (  # type: ignore[no-redef]
        ActivationExecutionError,
        ScopedActivationHooks,
        activation_observation,
        resolve_final_circuit_modules,
    )
    from inspect_model import sha256_argument  # type: ignore[no-redef]
    from intervention_schema import (  # type: ignore[no-redef]
        REPORT_SCHEMA_VERSION,
        derive_intervention_comparison,
        load_intervention_spec,
    )
    from probe_schema import ProbeSchemaError, load_probe_manifest, read_bounded_regular_file  # type: ignore[no-redef]


SANDBOX_PROFILE = "linux-userns-landlock-intervention-v1"
MAX_SPEC_BYTES = 4 * 1024 * 1024


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _h192_transform(item: dict[str, Any], strength: float, torch: Any):
    vector = torch.tensor(item["vector"], dtype=torch.float32)

    def transform(_module: Any, _inputs: Any, output: Any) -> Any:
        if not isinstance(output, torch.Tensor) or list(output.shape) != [192]:
            raise ActivationExecutionError("Intervention hook received an invalid h192 tensor.")
        target = vector.to(device=output.device, dtype=output.dtype)
        if item["mode"] == "add":
            return output + strength * target
        return (1.0 - strength) * output + strength * target

    return transform


def _run_once(
    model: Any,
    modules: Any,
    torch: Any,
    *,
    input_text: str,
    repetition: int,
    intervention: dict[str, Any] | None = None,
    strength: float = 0.0,
) -> tuple[dict[str, Any], bool]:
    transform = _h192_transform(intervention, strength, torch) if intervention else None
    hooks = ScopedActivationHooks(modules, h192_transform=transform)
    with hooks:
        hooks.begin_inference()
        output = model(input_text)
        observation = activation_observation(
            input_text=input_text,
            repetition=repetition,
            returned_output=output,
            captured=hooks.captured(),
            modules=modules,
            torch=torch,
        )
    return observation, hooks.hooks_removed


def run_interventions(
    model: Any,
    torch: Any,
    manifest: dict[str, Any],
    spec: dict[str, Any],
    repetitions: int,
) -> list[dict[str, Any]]:
    modules = resolve_final_circuit_modules(model, torch)
    by_case: dict[str, list[dict[str, Any]]] = {
        case["id"]: [item for item in spec["interventions"] if case["id"] in item["case_ids"]]
        for case in manifest["cases"]
    }
    results = []
    with torch.inference_mode():
        if torch.is_grad_enabled():
            raise ActivationExecutionError("Gradient tracking remained enabled in inference mode.")
        for case in manifest["cases"]:
            baseline = []
            groups = [
                {
                    "intervention_id": item["id"],
                    "strength": strength,
                    "observations": [],
                    "deterministic": False,
                }
                for item in by_case[case["id"]]
                for strength in item["strengths"]
            ]
            group_items = []
            position = 0
            for item in by_case[case["id"]]:
                for strength in item["strengths"]:
                    group_items.append((item, strength, groups[position]))
                    position += 1
            for repetition in range(repetitions):
                base, _hooks_removed = _run_once(
                    model, modules, torch, input_text=case["input"], repetition=repetition
                )
                baseline.append(base)
                for item, strength, group in group_items:
                    activated, _hooks_removed = _run_once(
                        model,
                        modules,
                        torch,
                        input_text=case["input"],
                        repetition=repetition,
                        intervention=item,
                        strength=float(strength),
                    )
                    group["observations"].append(
                        {
                            "activation": activated,
                            "comparison": derive_intervention_comparison(base, activated),
                        }
                    )
            for group in groups:
                signatures = [
                    item["activation"]["tensors"]["h192"]["float32_sha256"]
                    for item in group["observations"]
                ]
                group["deterministic"] = len(set(signatures)) == 1
            results.append({"case": case, "baseline": baseline, "interventions": groups})
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--expected-sha256", type=sha256_argument, required=True)
    parser.add_argument("--expected-manifest-sha256", type=sha256_argument, required=True)
    parser.add_argument("--expected-spec-sha256", type=sha256_argument, required=True)
    parser.add_argument("--repetitions", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if os.environ.get("JSMI_SANDBOX_PROFILE") != SANDBOX_PROFILE:
        print("Intervention worker refuses to run outside the required sandbox.", file=sys.stderr)
        return 2
    if not 2 <= args.repetitions <= 10:
        print("Intervention repetitions must be between 2 and 10.", file=sys.stderr)
        return 2
    actual_sha256 = sha256_file(args.model)
    if actual_sha256 != args.expected_sha256:
        print("Artifact hash changed inside the sandbox; refusing to unpickle.", file=sys.stderr)
        return 2
    try:
        manifest, _payload, manifest_sha256 = load_probe_manifest(args.manifest)
        spec_payload = read_bounded_regular_file(args.spec, MAX_SPEC_BYTES)
        spec, spec_sha256 = load_intervention_spec(
            spec_payload, case_ids={case["id"] for case in manifest["cases"]}
        )
    except ProbeSchemaError as exc:
        print(f"Intervention input rejected inside sandbox: {exc}", file=sys.stderr)
        return 2
    if manifest_sha256 != args.expected_manifest_sha256 or spec_sha256 != args.expected_spec_sha256:
        print("Intervention input hash changed inside the sandbox.", file=sys.stderr)
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
        results = run_interventions(model, torch, manifest, spec, args.repetitions)
    except ActivationExecutionError as exc:
        print(f"Intervention execution failed: {exc}", file=sys.stderr)
        return 1
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact": {"sha256": actual_sha256, "filename": args.artifact_name, "source_url": args.source_url},
        "probe_suite": {"suite_id": manifest["suite_id"], "manifest_sha256": manifest_sha256, "repetitions": args.repetitions},
        "source_activation_report": {"sha256": spec["source_activation_report_sha256"]},
        "intervention_spec": {"suite_id": spec["suite_id"], "sha256": spec_sha256},
        "execution": {"inference_mode": True, "model_eval": model.training is False, "hooks_removed": True},
        "results": results,
        "summary": {"intervention_observation_count": sum(len(group["observations"]) for result in results for group in result["interventions"])},
        "inference_executed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(f"{args.output.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(f"Intervention report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
