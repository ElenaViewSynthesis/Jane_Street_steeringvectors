from __future__ import annotations

import contextlib
import math
import unittest

from scripts.probe_worker import ProbeExecutionError, run_probe_cases, scalar_observation
from tests.test_probe_schema import valid_manifest


class FakeTensor:
    def __init__(self, value: float, shape: tuple[int, ...] = (1,)) -> None:
        self.value = value
        self.shape = shape
        self.dtype = "torch.float32"
        self.device = "cpu"

    def numel(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result

    def detach(self):
        return self

    def cpu(self):
        return self

    def reshape(self, *_shape):
        return self

    def __getitem__(self, _index):
        return self

    def item(self) -> float:
        return self.value


class FakeTorch:
    Tensor = FakeTensor

    @staticmethod
    def inference_mode():
        return contextlib.nullcontext()

    @staticmethod
    def is_grad_enabled() -> bool:
        return False


class ProbeWorkerTests(unittest.TestCase):
    def test_runs_the_complete_suite_in_each_repetition(self) -> None:
        calls: list[str] = []

        def model(input_text: str) -> FakeTensor:
            calls.append(input_text)
            return FakeTensor(float(len(input_text)))

        manifest = valid_manifest()
        results = run_probe_cases(model, FakeTorch, manifest, repetitions=2)

        self.assertEqual(calls, ["alpha", "beta", "alpha", "beta"])
        self.assertTrue(all(result["deterministic"] for result in results))
        self.assertEqual(results[0]["observations"][0]["value"], 5.0)
        self.assertEqual(results[1]["observations"][1]["float_hex"], 4.0.hex())

    def test_marks_stateful_output_as_nondeterministic(self) -> None:
        counter = 0

        def model(_input_text: str) -> FakeTensor:
            nonlocal counter
            counter += 1
            return FakeTensor(float(counter))

        results = run_probe_cases(model, FakeTorch, valid_manifest(), repetitions=2)

        self.assertTrue(all(not result["deterministic"] for result in results))

    def test_rejects_non_tensor_output(self) -> None:
        with self.assertRaisesRegex(ProbeExecutionError, "not a torch.Tensor"):
            scalar_observation(1.0, FakeTorch, repetition=0)

    def test_rejects_non_scalar_tensor_output(self) -> None:
        with self.assertRaisesRegex(ProbeExecutionError, "one scalar"):
            scalar_observation(FakeTensor(1.0, shape=(2,)), FakeTorch, repetition=0)

    def test_rejects_nonfinite_output(self) -> None:
        for value in (math.inf, -math.inf, math.nan):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ProbeExecutionError, "non-finite"):
                    scalar_observation(FakeTensor(value), FakeTorch, repetition=0)


if __name__ == "__main__":
    unittest.main()
