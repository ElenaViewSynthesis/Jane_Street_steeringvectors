from __future__ import annotations

import argparse
import hashlib
import json
import pickletools
import re
import sys
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_FILENAME = "model_3_11.pt" if sys.version_info >= (3, 11) else "model.pt"
DEFAULT_MODEL = ROOT / DEFAULT_MODEL_FILENAME
DEFAULT_REPORT = ROOT / "outputs" / "reports" / "archive_metadata.json"
DEFAULT_SOURCE_URL = (
    f"https://huggingface.co/jane-street/2025-03-10/resolve/main/{DEFAULT_MODEL_FILENAME}"
)
KNOWN_MODEL_SHA256 = {
    "model_3_11.pt": "43aa7da7ccf749ae1fb95f8b7a6aa49536b73e27f0ac74cb90d5f824ccd484b2",
}
HASH_CHUNK_SIZE = 8 * 1024 * 1024


def sha256_file(path: Path, chunk_size: int = HASH_CHUNK_SIZE) -> str:
    """Return a streamed SHA-256 digest without loading the artifact into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_artifact(
    model_path: Path,
    source_url: str | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    actual_sha256 = sha256_file(model_path)
    normalized_expected = expected_sha256.lower() if expected_sha256 else None
    return {
        "filename": model_path.name,
        "path": str(model_path),
        "size_bytes": model_path.stat().st_size,
        "sha256": actual_sha256,
        "expected_sha256": normalized_expected,
        "checksum_matches_expected": (
            actual_sha256 == normalized_expected if normalized_expected else None
        ),
        "source_url": source_url,
    }


def sha256_argument(value: str) -> str:
    normalized = value.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise argparse.ArgumentTypeError("SHA-256 must contain exactly 64 hexadecimal characters.")
    return normalized


def trusted_sha256_for(model_path: Path, supplied_digest: str | None = None) -> str | None:
    """Prefer an explicit digest, otherwise use a known official artifact digest."""
    return supplied_digest or KNOWN_MODEL_SHA256.get(model_path.name)


def read_text_file(zf: zipfile.ZipFile, name: str) -> str | None:
    try:
        return zf.read(name).decode("utf-8", errors="replace").strip()
    except KeyError:
        return None


def _global_name_from_arg(arg: Any) -> str:
    raw = str(arg)
    if "\n" in raw:
        module, name = raw.split("\n", 1)
    elif " " in raw:
        module, name = raw.split(" ", 1)
    else:
        return raw
    return f"{module}.{name}"


def _pop_marked_items(stack: list[Any], mark: object) -> list[Any]:
    idx = len(stack) - 1
    while idx >= 0 and stack[idx] is not mark:
        idx -= 1
    if idx < 0:
        raise ValueError("Missing MARK while decoding pickle stream.")
    items = stack[idx + 1 :]
    del stack[idx:]
    return items


def summarize_pickle_globals(
    pickle_bytes: bytes,
    storage_sizes_by_key: dict[str, int] | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    globals_found: list[str] = []
    opcode_counts: dict[str, int] = {}
    unicode_literals: list[str] = []
    persistent_ids: list[Any] = []
    mark = object()
    stack: list[Any] = []
    memo: dict[int, Any] = {}

    for opcode, arg, _pos in pickletools.genops(pickle_bytes):
        name = opcode.name
        opcode_counts[name] = opcode_counts.get(name, 0) + 1

        try:
            if name in ("BINUNICODE", "SHORT_BINUNICODE", "UNICODE"):
                stack.append(arg)
                if isinstance(arg, str):
                    unicode_literals.append(arg)
                continue

            if name == "GLOBAL":
                global_name = _global_name_from_arg(arg)
                globals_found.append(global_name)
                stack.append(("GLOBAL", global_name))
                continue

            if name == "STACK_GLOBAL":
                attr_name = stack.pop()
                module_name = stack.pop()
                global_name = f"{module_name}.{attr_name}"
                globals_found.append(global_name)
                stack.append(("GLOBAL", global_name))
                continue

            if name in ("BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4"):
                stack.append(arg)
                continue

            if name == "NONE":
                stack.append(None)
                continue

            if name == "NEWTRUE":
                stack.append(True)
                continue

            if name == "NEWFALSE":
                stack.append(False)
                continue

            if name == "MARK":
                stack.append(mark)
                continue

            if name == "EMPTY_TUPLE":
                stack.append(())
                continue

            if name == "EMPTY_LIST":
                stack.append([])
                continue

            if name == "EMPTY_DICT":
                stack.append({})
                continue

            if name == "TUPLE":
                stack.append(tuple(_pop_marked_items(stack, mark)))
                continue

            if name == "TUPLE1":
                stack.append((stack.pop(),))
                continue

            if name == "TUPLE2":
                right = stack.pop()
                left = stack.pop()
                stack.append((left, right))
                continue

            if name == "TUPLE3":
                third = stack.pop()
                second = stack.pop()
                first = stack.pop()
                stack.append((first, second, third))
                continue

            if name in ("BINPUT", "LONG_BINPUT"):
                memo[int(arg)] = stack[-1]
                continue

            if name == "PUT":
                memo[int(str(arg))] = stack[-1]
                continue

            if name in ("BINGET", "LONG_BINGET"):
                stack.append(memo[int(arg)])
                continue

            if name == "GET":
                stack.append(memo[int(str(arg))])
                continue

            if name == "SETITEM":
                value = stack.pop()
                key = stack.pop()
                dictionary = stack[-1]
                if isinstance(dictionary, dict):
                    dictionary[key] = value
                continue

            if name == "SETITEMS":
                items = _pop_marked_items(stack, mark)
                dictionary = stack[-1]
                if isinstance(dictionary, dict):
                    for idx in range(0, len(items), 2):
                        if idx + 1 < len(items):
                            dictionary[items[idx]] = items[idx + 1]
                continue

            if name == "APPEND":
                item = stack.pop()
                target = stack[-1]
                if isinstance(target, list):
                    target.append(item)
                continue

            if name == "APPENDS":
                items = _pop_marked_items(stack, mark)
                target = stack[-1]
                if isinstance(target, list):
                    target.extend(items)
                continue

            if name == "REDUCE":
                reduce_args = stack.pop()
                reduce_fn = stack.pop()
                stack.append(("REDUCE", reduce_fn, reduce_args))
                continue

            if name == "NEWOBJ":
                newobj_args = stack.pop()
                newobj_cls = stack.pop()
                stack.append(("NEWOBJ", newobj_cls, newobj_args))
                continue

            if name == "BUILD":
                state = stack.pop()
                obj = stack.pop()
                stack.append(("BUILD", obj, state))
                continue

            if name == "BINPERSID":
                pid = stack.pop()
                persistent_ids.append(pid)
                stack.append(("PERSISTENT", pid))
                continue

            if name == "POP":
                stack.pop()
                continue

            if name == "STOP":
                break
        except (IndexError, KeyError, ValueError):
            # Keep analysis best-effort and avoid failing on uncommon opcodes.
            continue

    unique_globals: list[str] = []
    seen: set[str] = set()
    for item in globals_found:
        if item not in seen:
            seen.add(item)
            unique_globals.append(item)
        if len(unique_globals) >= limit:
            break

    storage_sizes_by_key = storage_sizes_by_key or {}
    storage_refs: list[dict[str, Any]] = []
    unmatched_storage_keys: list[str] = []
    seen_storage_keys: set[str] = set()
    storage_type_counts: dict[str, int] = {}
    device_counts: dict[str, int] = {}

    for pid in persistent_ids:
        if not (isinstance(pid, tuple) and len(pid) >= 5 and pid[0] == "storage"):
            continue

        storage_type = pid[1]
        if isinstance(storage_type, tuple) and len(storage_type) >= 2 and storage_type[0] == "GLOBAL":
            storage_type_name = str(storage_type[1])
        else:
            storage_type_name = str(storage_type)

        key = str(pid[2])
        device = str(pid[3])
        numel = int(pid[4]) if isinstance(pid[4], int) else None
        bytes_in_archive = storage_sizes_by_key.get(key)
        bytes_per_item = (
            bytes_in_archive / numel
            if bytes_in_archive is not None and numel and numel > 0
            else None
        )

        storage_type_counts[storage_type_name] = storage_type_counts.get(storage_type_name, 0) + 1
        device_counts[device] = device_counts.get(device, 0) + 1
        seen_storage_keys.add(key)

        if bytes_in_archive is None:
            unmatched_storage_keys.append(key)

        storage_refs.append(
            {
                "key": key,
                "storage_type": storage_type_name,
                "device": device,
                "numel": numel,
                "bytes_in_archive": bytes_in_archive,
                "bytes_per_item": bytes_per_item,
            }
        )

    candidate_name_pattern = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{1,200}$")
    tensor_name_hints = [
        s
        for s in unicode_literals
        if candidate_name_pattern.match(s)
        and any(token in s for token in ("weight", "bias", "running_", "num_batches", "layer", "module"))
    ]
    unique_tensor_name_hints: list[str] = []
    seen_hints: set[str] = set()
    for hint in tensor_name_hints:
        if hint not in seen_hints:
            seen_hints.add(hint)
            unique_tensor_name_hints.append(hint)
        if len(unique_tensor_name_hints) >= 40:
            break

    return {
        "pickle_size_bytes": len(pickle_bytes),
        "unique_global_refs_sample": unique_globals,
        "opcode_counts": dict(sorted(opcode_counts.items())),
        "contains_stack_global": "STACK_GLOBAL" in opcode_counts,
        "persistent_id_count": len(persistent_ids),
        "storage_reference_count": len(storage_refs),
        "unique_storage_key_count": len(seen_storage_keys),
        "storage_type_counts": dict(sorted(storage_type_counts.items())),
        "storage_device_counts": dict(sorted(device_counts.items())),
        "storage_reference_sample": storage_refs[:limit],
        "unmatched_storage_keys_sample": unmatched_storage_keys[:limit],
        "storage_entries_without_reference_sample": sorted(
            set(storage_sizes_by_key.keys()) - seen_storage_keys
        )[:limit],
        "string_literal_count": len(unicode_literals),
        "tensor_name_hints_sample": unique_tensor_name_hints,
    }


def summarize_archive(
    model_path: Path,
    source_url: str | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    artifact = summarize_artifact(
        model_path,
        source_url=source_url,
        expected_sha256=expected_sha256,
    )
    with zipfile.ZipFile(model_path) as zf:
        names = zf.namelist()
        if not names:
            raise ValueError("Archive is empty.")

        root_prefix = names[0].split("/")[0]
        data_pickle_name = next((name for name in names if name.endswith("/data.pkl")), None)
        storage_entries = [
            name
            for name in names
            if name.startswith(f"{root_prefix}/data/") and not name.endswith("/")
        ]
        storage_sizes_by_key = {
            name.rsplit("/", 1)[-1]: zf.getinfo(name).file_size for name in storage_entries
        }
        infos = [zf.getinfo(name) for name in names if not name.endswith("/")]
        largest_entries = sorted(infos, key=lambda info: info.file_size, reverse=True)[:10]

        summary: dict[str, Any] = {
            "model_path": str(model_path),
            "artifact": artifact,
            "archive_format": "zip",
            "root_prefix": root_prefix,
            "entry_count": len(names),
            "storage_entry_count": len(storage_entries),
            "byteorder": read_text_file(zf, f"{root_prefix}/byteorder"),
            "version": read_text_file(zf, f"{root_prefix}/version"),
            "data_pickle_name": data_pickle_name,
            "data_pickle_summary": None,
            "largest_entries": [
                {"name": info.filename, "size_bytes": info.file_size}
                for info in largest_entries
            ],
            "sample_entries": names[:20],
        }

        if data_pickle_name is not None:
            summary["data_pickle_summary"] = summarize_pickle_globals(
                zf.read(data_pickle_name),
                storage_sizes_by_key=storage_sizes_by_key,
            )

        return summary


def print_archive_summary(summary: dict[str, Any]) -> None:
    artifact = summary["artifact"]
    print("Archive metadata")
    print(f"- model path: {summary['model_path']}")
    print(f"- size: {artifact['size_bytes']} bytes")
    print(f"- SHA-256: {artifact['sha256']}")
    if artifact.get("source_url"):
        print(f"- source: {artifact['source_url']}")
    if artifact.get("expected_sha256"):
        print(
            f"- expected SHA-256 matches: "
            f"{artifact['checksum_matches_expected']}"
        )
    print(f"- archive format: {summary['archive_format']}")
    print(f"- root prefix: {summary['root_prefix']}")
    print(f"- total entries: {summary['entry_count']}")
    print(f"- tensor storage entries: {summary['storage_entry_count']}")
    print(f"- byteorder: {summary['byteorder']}")
    print(f"- version: {summary['version']}")
    print(f"- pickle metadata file: {summary['data_pickle_name']}")

    data_pickle_summary = summary.get("data_pickle_summary") or {}
    if data_pickle_summary:
        print(f"- pickle bytes: {data_pickle_summary['pickle_size_bytes']}")
        print(
            f"- contains STACK_GLOBAL opcodes: "
            f"{data_pickle_summary['contains_stack_global']}"
        )
        print(f"- persistent IDs in pickle: {data_pickle_summary['persistent_id_count']}")
        print(
            f"- storage refs in pickle: {data_pickle_summary['storage_reference_count']} "
            f"(unique keys: {data_pickle_summary['unique_storage_key_count']})"
        )
        if data_pickle_summary.get("storage_device_counts"):
            print(
                f"- storage devices: "
                f"{json.dumps(data_pickle_summary['storage_device_counts'])}"
            )
        if data_pickle_summary.get("storage_type_counts"):
            print(
                f"- storage types: {json.dumps(data_pickle_summary['storage_type_counts'])}"
            )
        if data_pickle_summary.get("tensor_name_hints_sample"):
            print("- tensor name hints (sample):")
            for item in data_pickle_summary["tensor_name_hints_sample"][:10]:
                print(f"  - {item}")
        sample = data_pickle_summary.get("unique_global_refs_sample") or []
        if sample:
            print("- sample pickle GLOBAL refs:")
            for item in sample:
                print(f"  - {item}")
        storage_sample = data_pickle_summary.get("storage_reference_sample") or []
        if storage_sample:
            print("- sample storage mappings from data.pkl -> archive bytes:")
            for item in storage_sample[:5]:
                print(
                    "  - "
                    f"data/{item['key']}: numel={item['numel']}, "
                    f"bytes={item['bytes_in_archive']}, "
                    f"dtype_size~={item['bytes_per_item']}"
                )

    print("- largest archive members:")
    for item in summary["largest_entries"]:
        print(f"  - {item['name']} ({item['size_bytes']} bytes)")


def tensor_metadata(name: str, tensor: Any) -> dict[str, Any]:
    metadata = {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": tensor.numel(),
    }
    if hasattr(tensor, "requires_grad"):
        metadata["requires_grad"] = bool(tensor.requires_grad)
    return metadata


def module_metadata(name: str, module: Any) -> dict[str, Any]:
    direct_parameters = list(module.named_parameters(recurse=False))
    direct_buffers = list(module.named_buffers(recurse=False))
    return {
        "name": name or "<root>",
        "type": f"{type(module).__module__}.{type(module).__qualname__}",
        "direct_parameter_count": sum(tensor.numel() for _name, tensor in direct_parameters),
        "direct_buffer_count": sum(tensor.numel() for _name, tensor in direct_buffers),
        "direct_parameters": [
            tensor_metadata(parameter_name, tensor)
            for parameter_name, tensor in direct_parameters
        ],
        "direct_buffers": [
            tensor_metadata(buffer_name, tensor)
            for buffer_name, tensor in direct_buffers
        ],
    }


def summarize_loaded_object(obj: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "python_type": f"{type(obj).__module__}.{type(obj).__qualname__}",
    }

    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - handled by caller
        raise RuntimeError("torch is required for the controlled load phase.") from exc

    if isinstance(obj, nn.Module):
        children = list(obj.named_children())
        modules = list(obj.named_modules())
        named_modules = [(name, mod) for name, mod in modules if name]
        leaf_modules = [
            (name, mod)
            for name, mod in modules
            if not any(mod.children())
        ]
        params = list(obj.named_parameters())
        buffers = list(obj.named_buffers())

        summary.update(
            {
                "kind": "nn.Module",
                "module_class": obj.__class__.__name__,
                "training": bool(obj.training),
                "top_level_children": [name for name, _child in children],
                "module_tree": [
                    module_metadata(name, mod)
                    for name, mod in modules
                ],
                "last_two_named_modules": [
                    {"name": name, "type": mod.__class__.__name__}
                    for name, mod in named_modules[-2:]
                ],
                "final_two_leaf_modules": [
                    module_metadata(name, mod)
                    for name, mod in leaf_modules[-2:]
                ],
                "parameter_count": sum(param.numel() for _name, param in params),
                "buffer_count": sum(buf.numel() for _name, buf in buffers),
                "parameters": [
                    tensor_metadata(name, param)
                    for name, param in params
                ],
                "buffers": [
                    tensor_metadata(name, buffer)
                    for name, buffer in buffers
                ],
                "last_parameter_tensors": [
                    tensor_metadata(name, param)
                    for name, param in params[-6:]
                ],
                "state_dict_keys": list(obj.state_dict().keys()),
            }
        )
        return summary

    if isinstance(obj, dict):
        tensor_items = []
        non_tensor_keys = []
        for key, value in obj.items():
            if isinstance(value, torch.Tensor):
                tensor_items.append(
                    {
                        "key": str(key),
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                    }
                )
            else:
                non_tensor_keys.append({"key": str(key), "type": type(value).__name__})

        summary.update(
            {
                "kind": "dict",
                "key_count": len(obj),
                "tensor_value_count": len(tensor_items),
                "tensor_value_sample": tensor_items[:20],
                "non_tensor_value_sample": non_tensor_keys[:20],
            }
        )
        return summary

    summary.update(
        {
            "kind": "other",
            "repr": repr(obj)[:500],
            "attribute_sample": [name for name in dir(obj) if not name.startswith("_")][:30],
        }
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a PyTorch .pt file safely. By default this only reads archive metadata "
            "without unpickling the model."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Path to the model artifact. Default for this interpreter: {DEFAULT_MODEL.name}.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to write the archive metadata JSON report.",
    )
    parser.add_argument(
        "--source-url",
        default=DEFAULT_SOURCE_URL,
        help=(
            "Artifact source recorded in the report. Defaults to the Jane Street "
            "Hugging Face model URL."
        ),
    )
    parser.add_argument(
        "--expected-sha256",
        type=sha256_argument,
        default=None,
        help=(
            "Trusted SHA-256 override. Known official artifacts are verified automatically. "
            "A mismatch stops before any pickle execution."
        ),
    )
    parser.add_argument(
        "--load-pickle",
        action="store_true",
        help=(
            "Deprecated and disabled. Use scripts/run_architecture_sandbox.py for "
            "hardened live loading."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model.resolve()
    expected_sha256 = trusted_sha256_for(model_path, args.expected_sha256)

    if not model_path.exists():
        print(f"Model file not found: {model_path}", file=sys.stderr)
        return 1

    try:
        summary = summarize_archive(
            model_path,
            source_url=args.source_url,
            expected_sha256=expected_sha256,
        )
    except zipfile.BadZipFile as exc:
        print(f"Archive inspection failed: {exc}", file=sys.stderr)
        return 1

    print_archive_summary(summary)

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"- wrote report: {args.report}")

    if summary["artifact"]["checksum_matches_expected"] is False:
        print("Artifact SHA-256 does not match the trusted checksum.", file=sys.stderr)
        return 2

    if args.load_pickle:
        print(
            "Direct pickle loading is disabled. Use scripts/run_architecture_sandbox.py.",
            file=sys.stderr,
        )
        return 2

    print()
    print("No pickle payload was executed.")
    print("Use scripts/run_architecture_sandbox.py for hardened architecture loading.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
