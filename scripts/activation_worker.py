from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

try:
    from scripts.activation_schema import (
        CAPTURE_MODULES,
        CAPTURE_PROFILE,
        CAPTURE_SHAPES,
        PREDICATE_TARGETS,
        REPORT_SCHEMA_VERSION,
        SANDBOX_PROFILE,
        activation_signature,
        derive_activation_values,
        float32_sha256,
        summarize_activation_results,
    )
    from scripts.inspect_model import sha256_argument
    from scripts.probe_schema import ProbeSchemaError, load_probe_manifest
except ModuleNotFoundError:  # Direct execution from scripts/.
    from activation_schema import (  # type: ignore[no-redef]
        CAPTURE_MODULES,
        CAPTURE_PROFILE,
        CAPTURE_SHAPES,
        PREDICATE_TARGETS,
        REPORT_SCHEMA_VERSION,
        SANDBOX_PROFILE,
        activation_signature,
        derive_activation_values,
        float32_sha256,
        summarize_activation_results,
    )
    from inspect_model import sha256_argument  # type: ignore[no-redef]
    from probe_schema import ProbeSchemaError, load_probe_manifest  # type: ignore[no-redef]


class ActivationExecutionError(RuntimeError):
    """Raised when final-layer capture or verification violates its contract."""


@dataclass(frozen=True)
class FinalCircuitModules:
    h_relu: Any
    predicate_linear: Any
    predicate_relu: Any
    readout_linear: Any
    output_relu: Any

    def by_role(self) -> dict[str, Any]:
        return {
            "h192": self.h_relu,
            "a48": self.predicate_linear,
            "z48": self.predicate_relu,
            "readout": self.readout_linear,
            "output": self.output_relu,
        }


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _module_type(module: Any) -> str:
    return f"{type(module).__module__}.{type(module).__qualname__}"


def resolve_final_circuit_modules(model: Any, torch: Any) -> FinalCircuitModules:
    if not isinstance(model, torch.nn.Module):
        raise ActivationExecutionError("Loaded object is not a torch.nn.Module.")
    leaves = [
        (name, module)
        for name, module in model.named_modules()
        if not any(module.children())
    ]
    if len(leaves) < len(CAPTURE_MODULES):
        raise ActivationExecutionError("Model does not contain the required final modules.")
    tail = leaves[-len(CAPTURE_MODULES) :]
    actual_metadata = [
        {
            "module_name": name,
            "module_type": _module_type(module),
        }
        for name, module in tail
    ]
    expected_metadata = [
        {
            "module_name": item["module_name"],
            "module_type": item["module_type"],
        }
        for item in CAPTURE_MODULES
    ]
    if actual_metadata != expected_metadata:
        raise ActivationExecutionError(
            "Final module names or types do not match the hash-bound architecture."
        )

    h_relu = tail[0][1]
    predicate_linear = tail[1][1]
    predicate_relu = tail[2][1]
    readout_linear = tail[3][1]
    output_relu = tail[4][1]
    if not isinstance(predicate_linear, torch.nn.Linear) or (
        predicate_linear.in_features,
        predicate_linear.out_features,
    ) != (192, 48):
        raise ActivationExecutionError("Predicate layer is not Linear(192, 48).")
    if not isinstance(readout_linear, torch.nn.Linear) or (
        readout_linear.in_features,
        readout_linear.out_features,
    ) != (48, 1):
        raise ActivationExecutionError("Readout layer is not Linear(48, 1).")

    predicate_bias = predicate_linear.bias.detach().cpu().reshape(-1).tolist()
    central_targets = [-float(value) for value in predicate_bias[16:32]]
    if central_targets != PREDICATE_TARGETS:
        raise ActivationExecutionError("Predicate targets do not match architecture recovery.")
    readout_weights = readout_linear.weight.detach().cpu().reshape(-1).tolist()
    if readout_weights != ([1.0] * 16 + [-2.0] * 16 + [1.0] * 16):
        raise ActivationExecutionError("Readout weights do not match the predicate circuit.")
    readout_bias = readout_linear.bias.detach().cpu().reshape(-1).tolist()
    if readout_bias != [-15.0]:
        raise ActivationExecutionError("Readout bias does not match the predicate circuit.")

    return FinalCircuitModules(
        h_relu=h_relu,
        predicate_linear=predicate_linear,
        predicate_relu=predicate_relu,
        readout_linear=readout_linear,
        output_relu=output_relu,
    )


class ScopedActivationHooks(AbstractContextManager["ScopedActivationHooks"]):
    def __init__(
        self,
        modules: FinalCircuitModules,
        *,
        h192_transform: Any | None = None,
    ) -> None:
        self._modules = modules
        self._h192_transform = h192_transform
        self._handles: list[Any] = []
        self._current: dict[str, Any] = {}
        self.hooks_removed = False

    def _capture(self, role: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if role in self._current:
                raise ActivationExecutionError(
                    f"Activation hook {role!r} fired more than once for one inference."
                )
            self._current[role] = output

        return hook

    def __enter__(self) -> "ScopedActivationHooks":
        if self._handles:
            raise ActivationExecutionError("Activation hooks are already registered.")
        if self._h192_transform is not None:
            self._handles.append(
                self._modules.h_relu.register_forward_hook(self._h192_transform)
            )
        for role, module in self._modules.by_role().items():
            self._handles.append(module.register_forward_hook(self._capture(role)))
        self.hooks_removed = False
        return self

    def begin_inference(self) -> None:
        if not self._handles:
            raise ActivationExecutionError("Activation hooks are not registered.")
        self._current.clear()

    def captured(self) -> dict[str, Any]:
        missing = set(CAPTURE_SHAPES) - set(self._current)
        extra = set(self._current) - set(CAPTURE_SHAPES)
        if missing or extra:
            raise ActivationExecutionError(
                "Activation hook coverage does not match the capture profile."
            )
        return dict(self._current)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        self._current.clear()
        self.hooks_removed = True
        return None


def tensor_snapshot(tensor: Any, torch: Any, *, role: str) -> dict[str, Any]:
    if not isinstance(tensor, torch.Tensor):
        raise ActivationExecutionError(f"Hook {role!r} did not capture a torch.Tensor.")
    detached = tensor.detach().cpu()
    expected_shape = CAPTURE_SHAPES[role]
    if list(detached.shape) != expected_shape:
        raise ActivationExecutionError(
            f"Hook {role!r} captured shape {list(detached.shape)}, expected {expected_shape}."
        )
    if detached.dtype != torch.float32:
        raise ActivationExecutionError(f"Hook {role!r} did not capture float32 values.")
    values = [float(value) for value in detached.reshape(-1).tolist()]
    if any(not math.isfinite(value) for value in values):
        raise ActivationExecutionError(f"Hook {role!r} captured non-finite values.")
    try:
        digest = float32_sha256(values)
    except ProbeSchemaError as exc:
        raise ActivationExecutionError(str(exc)) from exc
    return {
        "dtype": str(detached.dtype),
        "shape": list(detached.shape),
        "device": str(detached.device),
        "values": values,
        "float32_sha256": digest,
    }


def _max_abs_error(actual: Any, expected: Any) -> float:
    return float((actual.detach().cpu() - expected.detach().cpu()).abs().max().item())


def activation_observation(
    *,
    input_text: str,
    repetition: int,
    returned_output: Any,
    captured: dict[str, Any],
    modules: FinalCircuitModules,
    torch: Any,
) -> dict[str, Any]:
    tensors = {
        role: tensor_snapshot(captured[role], torch, role=role)
        for role in CAPTURE_SHAPES
    }
    h192 = captured["h192"].detach()
    a48 = captured["a48"].detach()
    z48 = captured["z48"].detach()
    readout = captured["readout"].detach()
    output = captured["output"].detach()

    expected_a48 = torch.nn.functional.linear(
        h192,
        modules.predicate_linear.weight,
        modules.predicate_linear.bias,
    )
    expected_z48 = torch.relu(a48)
    expected_readout = torch.nn.functional.linear(
        z48,
        modules.readout_linear.weight,
        modules.readout_linear.bias,
    )
    expected_output = torch.relu(readout)
    returned_matches = (
        isinstance(returned_output, torch.Tensor)
        and returned_output.shape == output.shape
        and torch.equal(returned_output.detach().cpu(), output.detach().cpu())
    )
    verification = {
        "a48_max_abs_error": _max_abs_error(a48, expected_a48),
        "z48_max_abs_error": _max_abs_error(z48, expected_z48),
        "readout_max_abs_error": _max_abs_error(readout, expected_readout),
        "output_max_abs_error": _max_abs_error(output, expected_output),
        "returned_output_matches_hook": returned_matches,
    }
    return {
        "repetition": repetition,
        "tensors": tensors,
        "derived": derive_activation_values(input_text, tensors),
        "verification": verification,
    }


def run_activation_cases(
    model: Any,
    torch: Any,
    manifest: dict[str, Any],
    repetitions: int,
) -> tuple[list[dict[str, Any]], bool]:
    modules = resolve_final_circuit_modules(model, torch)
    results = [
        {
            "case": case,
            "input_length": len(case["input"]),
            "observations": [],
            "deterministic": False,
        }
        for case in manifest["cases"]
    ]
    hooks = ScopedActivationHooks(modules)
    with hooks:
        with torch.inference_mode():
            if torch.is_grad_enabled():
                raise ActivationExecutionError(
                    "Gradient tracking remained enabled in inference mode."
                )
            for repetition in range(repetitions):
                for result in results:
                    case = result["case"]
                    hooks.begin_inference()
                    try:
                        returned_output = model(case["input"])
                        observation = activation_observation(
                            input_text=case["input"],
                            repetition=repetition,
                            returned_output=returned_output,
                            captured=hooks.captured(),
                            modules=modules,
                            torch=torch,
                        )
                    except Exception as exc:
                        raise ActivationExecutionError(
                            f"Activation case {case['id']!r} failed during repetition "
                            f"{repetition}."
                        ) from exc
                    result["observations"].append(observation)

    for result in results:
        result["deterministic"] = (
            len({activation_signature(item) for item in result["observations"]}) == 1
        )
    return results, hooks.hooks_removed


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
        print("Activation worker refuses to run outside the required sandbox.", file=sys.stderr)
        return 2
    if not 2 <= args.repetitions <= 10:
        print("Activation repetitions must be between 2 and 10.", file=sys.stderr)
        return 2

    actual_sha256 = sha256_file(args.model)
    if actual_sha256 != args.expected_sha256:
        print("Artifact hash changed inside the sandbox; refusing to unpickle.", file=sys.stderr)
        return 2
    try:
        manifest, _manifest_payload, manifest_sha256 = load_probe_manifest(args.manifest)
    except ProbeSchemaError as exc:
        print(f"Activation manifest rejected inside sandbox: {exc}", file=sys.stderr)
        return 2
    if manifest_sha256 != args.expected_manifest_sha256:
        print("Activation manifest hash changed inside the sandbox.", file=sys.stderr)
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
        results, hooks_removed = run_activation_cases(
            model,
            torch,
            manifest,
            args.repetitions,
        )
    except ActivationExecutionError as exc:
        print(f"Activation execution failed: {exc}", file=sys.stderr)
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
        "capture": {
            "profile": CAPTURE_PROFILE,
            "scoped_hooks": True,
            "hooks_removed": hooks_removed,
            "modules": CAPTURE_MODULES,
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
        "summary": summarize_activation_results(results),
        "inference_executed": True,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(f"{args.output.suffix}.tmp")
    temporary_output.write_text(
        json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_output, args.output)
    print(f"Activation report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
