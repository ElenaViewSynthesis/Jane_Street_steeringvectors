from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from scripts.inspect_model import (
    DEFAULT_MODEL,
    main,
    sha256_argument,
    summarize_archive,
    summarize_loaded_object,
    summarize_pickle_globals,
    trusted_sha256_for,
)
from tests.fixtures import stack_global_pickle, storage_reference_pickle, write_tiny_torch_archive


class PickleSummaryTests(unittest.TestCase):
    def test_maps_persistent_storage_reference_to_archive_bytes(self) -> None:
        summary = summarize_pickle_globals(
            storage_reference_pickle(),
            storage_sizes_by_key={"0": 16},
        )

        self.assertEqual(summary["persistent_id_count"], 1)
        self.assertEqual(summary["storage_reference_count"], 1)
        self.assertEqual(summary["unique_storage_key_count"], 1)
        self.assertEqual(summary["storage_type_counts"], {"torch.FloatStorage": 1})
        self.assertEqual(summary["storage_device_counts"], {"cpu": 1})
        self.assertEqual(
            summary["storage_reference_sample"][0],
            {
                "key": "0",
                "storage_type": "torch.FloatStorage",
                "device": "cpu",
                "numel": 4,
                "bytes_in_archive": 16,
                "bytes_per_item": 4.0,
            },
        )

    def test_handles_stack_global_without_importing_target_module(self) -> None:
        summary = summarize_pickle_globals(stack_global_pickle())

        self.assertTrue(summary["contains_stack_global"])
        self.assertEqual(summary["unique_global_refs_sample"], ["torch.FloatStorage"])


class ArchiveSummaryTests(unittest.TestCase):
    def test_summarizes_fixture_checksum_provenance_and_storage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = write_tiny_torch_archive(Path(directory) / "model.pt")
            expected_digest = hashlib.sha256(model_path.read_bytes()).hexdigest()

            summary = summarize_archive(
                model_path,
                source_url="https://example.test/model.pt",
                expected_sha256=expected_digest.upper(),
            )

        self.assertEqual(summary["archive_format"], "zip")
        self.assertEqual(summary["root_prefix"], "puzzle")
        self.assertEqual(summary["storage_entry_count"], 1)
        self.assertEqual(summary["byteorder"], "little")
        self.assertEqual(summary["version"], "3")
        self.assertEqual(summary["artifact"]["sha256"], expected_digest)
        self.assertTrue(summary["artifact"]["checksum_matches_expected"])
        self.assertEqual(summary["artifact"]["source_url"], "https://example.test/model.pt")

    def test_reports_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = write_tiny_torch_archive(Path(directory) / "model.pt")
            summary = summarize_archive(model_path, expected_sha256="0" * 64)

        self.assertFalse(summary["artifact"]["checksum_matches_expected"])

    def test_checksum_mismatch_prevents_opted_in_pickle_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = write_tiny_torch_archive(Path(directory) / "model.pt")
            arguments = [
                "inspect_model.py",
                "--model",
                str(model_path),
                "--expected-sha256",
                "0" * 64,
                "--load-pickle",
            ]
            with (
                mock.patch.object(sys, "argv", arguments),
                mock.patch("builtins.print"),
            ):
                return_code = main()

        self.assertEqual(return_code, 2)

    def test_direct_pickle_load_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = write_tiny_torch_archive(Path(directory) / "model.pt")
            arguments = [
                "inspect_model.py",
                "--model",
                str(model_path),
                "--load-pickle",
            ]
            with (
                mock.patch.object(sys, "argv", arguments),
                mock.patch("builtins.print"),
            ):
                return_code = main()

        self.assertEqual(return_code, 2)


class ArgumentValidationTests(unittest.TestCase):
    def test_selects_python_311_model_for_current_interpreter(self) -> None:
        expected_name = "model_3_11.pt" if sys.version_info >= (3, 11) else "model.pt"
        self.assertEqual(DEFAULT_MODEL.name, expected_name)

    def test_normalizes_valid_sha256(self) -> None:
        self.assertEqual(sha256_argument("A" * 64), "a" * 64)

    def test_uses_known_digest_for_python_311_artifact(self) -> None:
        self.assertEqual(
            trusted_sha256_for(Path("model_3_11.pt")),
            "43aa7da7ccf749ae1fb95f8b7a6aa49536b73e27f0ac74cb90d5f824ccd484b2",
        )

    def test_explicit_digest_overrides_known_digest(self) -> None:
        self.assertEqual(trusted_sha256_for(Path("model_3_11.pt"), "a" * 64), "a" * 64)

    def test_rejects_invalid_sha256(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            sha256_argument("not-a-digest")


class LoadedModelSummaryTests(unittest.TestCase):
    class FakeTensor:
        def __init__(
            self,
            shape: tuple[int, ...],
            *,
            dtype: str = "torch.float32",
            requires_grad: bool = True,
        ) -> None:
            self.shape = shape
            self.dtype = dtype
            self.requires_grad = requires_grad

        def numel(self) -> int:
            result = 1
            for dimension in self.shape:
                result *= dimension
            return result

    class FakeModule:
        def __init__(
            self,
            *,
            children: dict[str, "LoadedModelSummaryTests.FakeModule"] | None = None,
            parameters: dict[str, "LoadedModelSummaryTests.FakeTensor"] | None = None,
            buffers: dict[str, "LoadedModelSummaryTests.FakeTensor"] | None = None,
        ) -> None:
            self.training = False
            self._children = children or {}
            self._parameters = parameters or {}
            self._buffers = buffers or {}

        def named_children(self):
            return iter(self._children.items())

        def children(self):
            return iter(self._children.values())

        def named_modules(self, prefix: str = ""):
            yield prefix, self
            for child_name, child in self._children.items():
                child_prefix = f"{prefix}.{child_name}" if prefix else child_name
                yield from child.named_modules(child_prefix)

        def named_parameters(self, recurse: bool = True, prefix: str = ""):
            for name, tensor in self._parameters.items():
                yield (f"{prefix}.{name}" if prefix else name), tensor
            if recurse:
                for child_name, child in self._children.items():
                    child_prefix = f"{prefix}.{child_name}" if prefix else child_name
                    yield from child.named_parameters(prefix=child_prefix)

        def named_buffers(self, recurse: bool = True, prefix: str = ""):
            for name, tensor in self._buffers.items():
                yield (f"{prefix}.{name}" if prefix else name), tensor
            if recurse:
                for child_name, child in self._children.items():
                    child_prefix = f"{prefix}.{child_name}" if prefix else child_name
                    yield from child.named_buffers(prefix=child_prefix)

        def state_dict(self):
            return dict(self.named_parameters()) | dict(self.named_buffers())

    def test_exports_complete_module_and_parameter_metadata(self) -> None:
        encoder = self.FakeModule(
            parameters={"weight": self.FakeTensor((3, 4))},
        )
        head = self.FakeModule(
            parameters={
                "weight": self.FakeTensor((1, 3)),
                "bias": self.FakeTensor((1,)),
            },
            buffers={
                "scale": self.FakeTensor((1,), requires_grad=False),
            },
        )
        model = self.FakeModule(children={"encoder": encoder, "head": head})
        fake_torch = types.SimpleNamespace(
            Tensor=self.FakeTensor,
            nn=types.SimpleNamespace(Module=self.FakeModule),
        )

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            summary = summarize_loaded_object(model)

        self.assertEqual(summary["kind"], "nn.Module")
        self.assertEqual(summary["top_level_children"], ["encoder", "head"])
        self.assertEqual(summary["parameter_count"], 16)
        self.assertEqual(summary["buffer_count"], 1)
        self.assertEqual(
            [module["name"] for module in summary["module_tree"]],
            ["<root>", "encoder", "head"],
        )
        self.assertEqual(
            [module["name"] for module in summary["final_two_leaf_modules"]],
            ["encoder", "head"],
        )
        self.assertEqual(
            [parameter["name"] for parameter in summary["parameters"]],
            ["encoder.weight", "head.weight", "head.bias"],
        )
        self.assertEqual(
            summary["state_dict_keys"],
            ["encoder.weight", "head.weight", "head.bias", "head.scale"],
        )


if __name__ == "__main__":
    unittest.main()
