from __future__ import annotations

import argparse
import dis
import functools
import hashlib
import json
import os
import platform
import sys
import types
from pathlib import Path
from typing import Any

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

try:
    from scripts.inspect_model import sha256_argument, summarize_loaded_object
except ModuleNotFoundError:  # Direct execution from scripts/.
    from inspect_model import sha256_argument, summarize_loaded_object  # type: ignore[no-redef]


SANDBOX_PROFILE = "linux-userns-landlock-v1"
MAX_DISASSEMBLY_CHARS = 200_000
MAX_INLINE_TENSOR_VALUES = 16_384
TOP_ABSOLUTE_VALUES = 20
MAX_REPORTED_UNIQUE_VALUES = 128


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_constant(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:2_000]
    if isinstance(value, bytes):
        return {"type": "bytes", "length": len(value), "hex_prefix": value[:64].hex()}
    if isinstance(value, tuple):
        return [safe_constant(item) for item in value[:50]]
    if isinstance(value, types.CodeType):
        return {
            "type": "code",
            "name": value.co_name,
            "qualified_name": value.co_qualname,
            "first_line": value.co_firstlineno,
        }
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def code_metadata(code: types.CodeType) -> dict[str, Any]:
    rendered_disassembly = dis.Bytecode(code).dis()
    return {
        "name": code.co_name,
        "qualified_name": code.co_qualname,
        "filename": code.co_filename,
        "first_line": code.co_firstlineno,
        "argument_count": code.co_argcount,
        "positional_only_argument_count": code.co_posonlyargcount,
        "keyword_only_argument_count": code.co_kwonlyargcount,
        "local_count": code.co_nlocals,
        "flags": code.co_flags,
        "variable_names": list(code.co_varnames),
        "free_variables": list(code.co_freevars),
        "cell_variables": list(code.co_cellvars),
        "referenced_names": list(code.co_names),
        "constants": [safe_constant(value) for value in code.co_consts],
        "bytecode_sha256": hashlib.sha256(code.co_code).hexdigest(),
        "disassembly": rendered_disassembly[:MAX_DISASSEMBLY_CHARS],
        "disassembly_truncated": len(rendered_disassembly) > MAX_DISASSEMBLY_CHARS,
    }


def callable_metadata(label: str, value: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "label": label,
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
    }
    function: Any = value
    if isinstance(value, functools.partial):
        metadata["partial_argument_count"] = len(value.args)
        metadata["partial_keyword_names"] = sorted((value.keywords or {}).keys())
        function = value.func
    if isinstance(function, types.MethodType):
        function = function.__func__
    if isinstance(function, types.FunctionType):
        metadata.update(
            {
                "module": function.__module__,
                "name": function.__name__,
                "qualified_name": function.__qualname__,
                "defaults": safe_constant(function.__defaults__),
                "keyword_defaults": safe_constant(function.__kwdefaults__),
                "code": code_metadata(function.__code__),
            }
        )
        closure_values = []
        for name, cell in zip(function.__code__.co_freevars, function.__closure__ or ()):
            try:
                value_metadata = safe_constant(cell.cell_contents)
            except ValueError:
                value_metadata = {"type": "empty_cell"}
            closure_values.append({"name": name, "value": value_metadata})
        metadata["closure"] = closure_values
        metadata["referenced_global_types"] = {
            name: f"{type(function.__globals__[name]).__module__}."
            f"{type(function.__globals__[name]).__qualname__}"
            for name in function.__code__.co_names
            if name in function.__globals__
        }
    return metadata


def collect_module_callables(model: Any) -> list[dict[str, Any]]:
    discovered: list[tuple[str, Any]] = []
    root_forward = getattr(type(model), "forward", None)
    if isinstance(root_forward, (types.FunctionType, types.MethodType)):
        discovered.append(("<root>.forward", root_forward))

    if hasattr(model, "named_modules"):
        for module_name, module in model.named_modules():
            display_name = module_name or "<root>"
            for attribute_name, value in vars(module).items():
                if isinstance(value, (types.FunctionType, types.MethodType, functools.partial)):
                    discovered.append((f"{display_name}.{attribute_name}", value))
                elif isinstance(value, dict):
                    for key, item in value.items():
                        if isinstance(item, (types.FunctionType, types.MethodType, functools.partial)):
                            discovered.append(
                                (f"{display_name}.{attribute_name}[{safe_constant(key)}]", item)
                            )

    results: list[dict[str, Any]] = []
    seen: set[int] = set()
    for label, value in discovered:
        identity = id(value.func if isinstance(value, functools.partial) else value)
        if identity in seen:
            continue
        seen.add(identity)
        results.append(callable_metadata(label, value))
    return results


def tensor_statistics(name: str, tensor: Any, torch: Any) -> dict[str, Any]:
    detached = tensor.detach().cpu()
    flattened = detached.reshape(-1)
    numeric = flattened.to(dtype=torch.float64)
    finite_mask = torch.isfinite(numeric)
    finite_values = numeric[finite_mask]
    result: dict[str, Any] = {
        "name": name,
        "shape": list(detached.shape),
        "stride": list(detached.stride()),
        "storage_offset": detached.storage_offset(),
        "dtype": str(detached.dtype),
        "numel": detached.numel(),
        "finite_count": int(finite_mask.sum().item()),
        "nonfinite_count": int((~finite_mask).sum().item()),
    }
    if finite_values.numel():
        result.update(
            {
                "minimum": float(finite_values.min().item()),
                "maximum": float(finite_values.max().item()),
                "mean": float(finite_values.mean().item()),
                "standard_deviation": float(finite_values.std(unbiased=False).item()),
                "l1_norm": float(finite_values.abs().sum().item()),
                "l2_norm": float(torch.linalg.vector_norm(finite_values).item()),
                "zero_count": int((finite_values == 0).sum().item()),
            }
        )
        unique_values, unique_counts = torch.unique(
            finite_values,
            sorted=True,
            return_counts=True,
        )
        result["unique_value_count"] = unique_values.numel()
        if unique_values.numel() <= MAX_REPORTED_UNIQUE_VALUES:
            result["unique_value_counts"] = [
                {"value": float(value), "count": int(count)}
                for value, count in zip(unique_values.tolist(), unique_counts.tolist())
            ]
    finite_indices = torch.nonzero(finite_mask, as_tuple=False).reshape(-1)
    top_count = min(TOP_ABSOLUTE_VALUES, finite_indices.numel())
    if top_count:
        finite_numeric = numeric[finite_indices]
        _magnitudes, finite_positions = torch.topk(finite_numeric.abs(), k=top_count)
        indices = finite_indices[finite_positions]
        result["largest_absolute_values"] = [
            {"flat_index": int(index), "value": float(numeric[index].item())}
            for index in indices.tolist()
        ]
    if numeric.numel() <= MAX_INLINE_TENSOR_VALUES:
        result["values"] = detached.tolist()
    return result


def analyze_final_layers(model: Any, torch: Any) -> list[dict[str, Any]]:
    if not isinstance(model, torch.nn.Module):
        return []
    leaf_modules = [
        (name, module)
        for name, module in model.named_modules()
        if not any(module.children())
    ]
    analyses = []
    for name, module in leaf_modules[-2:]:
        item: dict[str, Any] = {
            "name": name or "<root>",
            "type": f"{type(module).__module__}.{type(module).__qualname__}",
            "parameters": [
                tensor_statistics(parameter_name, parameter, torch)
                for parameter_name, parameter in module.named_parameters(recurse=False)
            ],
            "buffers": [
                tensor_statistics(buffer_name, buffer, torch)
                for buffer_name, buffer in module.named_buffers(recurse=False)
            ],
        }
        if isinstance(module, torch.nn.Linear):
            item["linear"] = {
                "in_features": module.in_features,
                "out_features": module.out_features,
                "has_bias": module.bias is not None,
            }
        analyses.append(item)
    return analyses


def analyze_final_parameterized_layers(model: Any, torch: Any) -> list[dict[str, Any]]:
    if not isinstance(model, torch.nn.Module):
        return []
    parameterized_leaf_modules = [
        (name, module)
        for name, module in model.named_modules()
        if not any(module.children())
        and any(True for _name, _parameter in module.named_parameters(recurse=False))
    ]
    analyses = []
    for name, module in parameterized_leaf_modules[-2:]:
        item: dict[str, Any] = {
            "name": name or "<root>",
            "type": f"{type(module).__module__}.{type(module).__qualname__}",
            "parameters": [
                tensor_statistics(parameter_name, parameter, torch)
                for parameter_name, parameter in module.named_parameters(recurse=False)
            ],
            "buffers": [
                tensor_statistics(buffer_name, buffer, torch)
                for buffer_name, buffer in module.named_buffers(recurse=False)
            ],
        }
        if isinstance(module, torch.nn.Linear):
            item["linear"] = {
                "in_features": module.in_features,
                "out_features": module.out_features,
                "has_bias": module.bias is not None,
            }
        analyses.append(item)
    return analyses


def _parameter_values(layer: dict[str, Any], parameter_name: str) -> Any:
    for parameter in layer.get("parameters", []):
        if parameter.get("name") == parameter_name and "values" in parameter:
            return parameter["values"]
    raise ValueError(
        f"Layer {layer.get('name', '<unknown>')} does not expose {parameter_name} values."
    )


def derive_final_predicate_circuit(
    final_parameterized_layers: list[dict[str, Any]],
    *,
    block_width: int = 8,
) -> dict[str, Any]:
    if len(final_parameterized_layers) != 2:
        raise ValueError("Expected exactly two final parameterized layers.")
    predicate_layer, output_layer = final_parameterized_layers
    predicate_weights = _parameter_values(predicate_layer, "weight")
    predicate_biases = _parameter_values(predicate_layer, "bias")
    output_weights = _parameter_values(output_layer, "weight")
    output_biases = _parameter_values(output_layer, "bias")

    if len(output_weights) != 1 or len(output_biases) != 1:
        raise ValueError("Expected a scalar final linear layer.")
    flat_output_weights = output_weights[0]
    if len(flat_output_weights) % 3:
        raise ValueError("Final weight count is not divisible into three predicate groups.")
    predicate_count = len(flat_output_weights) // 3
    expected_output_weights = (
        [1.0] * predicate_count
        + [-2.0] * predicate_count
        + [1.0] * predicate_count
    )
    if flat_output_weights != expected_output_weights:
        raise ValueError("Final layer does not implement the expected [1, -2, 1] pattern.")
    if output_biases[0] != float(-(predicate_count - 1)):
        raise ValueError("Final bias does not implement an all-predicates threshold.")
    if len(predicate_weights) != 3 * predicate_count:
        raise ValueError("Predicate layer row count does not match the final layer.")
    if len(predicate_biases) != 3 * predicate_count:
        raise ValueError("Predicate bias count does not match the final layer.")

    predicates = []
    powers = [float(2**index) for index in range(block_width)]
    for index in range(predicate_count):
        rows = [
            predicate_weights[index],
            predicate_weights[predicate_count + index],
            predicate_weights[2 * predicate_count + index],
        ]
        if rows[0] != rows[1] or rows[1] != rows[2]:
            raise ValueError(f"Predicate {index} does not reuse one affine expression.")
        biases = [
            predicate_biases[index],
            predicate_biases[predicate_count + index],
            predicate_biases[2 * predicate_count + index],
        ]
        if not (biases[0] == biases[1] - 1 and biases[2] == biases[1] + 1):
            raise ValueError(f"Predicate {index} biases are not consecutive thresholds.")

        row = rows[0]
        if len(row) % block_width:
            raise ValueError("Predicate input width is not divisible by the bit-block width.")
        block_terms = []
        for block_index in range(len(row) // block_width):
            coefficients = row[
                block_index * block_width : (block_index + 1) * block_width
            ]
            if not any(coefficients):
                continue
            scales = [
                coefficient / power
                for coefficient, power in zip(coefficients, powers)
            ]
            if any(scale != scales[0] for scale in scales):
                raise ValueError(
                    f"Predicate {index}, block {block_index} is not bit-weighted."
                )
            block_terms.append({"block": block_index, "coefficient": scales[0]})

        predicates.append(
            {
                "index": index,
                "target": -biases[1],
                "block_terms": block_terms,
                "relu_biases": biases,
            }
        )

    return {
        "bit_block_width": block_width,
        "predicate_count": predicate_count,
        "predicates": predicates,
        "indicator_weights": [1.0, -2.0, 1.0],
        "final_bias": output_biases[0],
        "integer_semantics": (
            "Each three-ReLU finite difference is 1 exactly at its target and 0 "
            "at other integer values; the final threshold is positive only when "
            "all predicates hold."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--expected-sha256", type=sha256_argument, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sandbox_profile = os.environ.get("JSMI_SANDBOX_PROFILE")
    if sandbox_profile != SANDBOX_PROFILE:
        print("Architecture worker refuses to run outside the required sandbox.", file=sys.stderr)
        return 2

    actual_sha256 = sha256_file(args.model)
    if actual_sha256 != args.expected_sha256:
        print("Artifact hash changed inside the sandbox; refusing to unpickle.", file=sys.stderr)
        return 2

    import cloudpickle
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    model = torch.load(args.model, map_location="cpu", weights_only=False)
    if isinstance(model, torch.nn.Module):
        model.eval()

    final_two_leaf_layers = analyze_final_layers(model, torch)
    final_two_parameterized_layers = analyze_final_parameterized_layers(model, torch)
    final_predicate_circuit = derive_final_predicate_circuit(
        final_two_parameterized_layers
    )

    report = {
        "schema_version": 1,
        "artifact": {
            "filename": args.artifact_name,
            "size_bytes": args.model.stat().st_size,
            "sha256": actual_sha256,
            "source_url": args.source_url,
        },
        "sandbox": {
            "profile": sandbox_profile,
            "network_namespace": "isolated",
            "filesystem_policy": "Landlock allow-list",
            "model_access": "read-only",
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
        "loaded_object": summarize_loaded_object(model),
        "final_two_leaf_layers": final_two_leaf_layers,
        "final_two_parameterized_layers": final_two_parameterized_layers,
        "final_predicate_circuit": final_predicate_circuit,
        "callables": collect_module_callables(model),
        "inference_executed": False,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(f"{args.output.suffix}.tmp")
    temporary_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_output, args.output)
    print(f"Architecture report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
