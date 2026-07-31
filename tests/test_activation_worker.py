from __future__ import annotations

import unittest

from scripts.activation_worker import (
    ActivationExecutionError,
    FinalCircuitModules,
    ScopedActivationHooks,
)


class FakeHandle:
    def __init__(self, hooks: list, hook) -> None:
        self._hooks = hooks
        self._hook = hook

    def remove(self) -> None:
        self._hooks.remove(self._hook)


class FakeModule:
    def __init__(self) -> None:
        self.hooks: list = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return FakeHandle(self.hooks, hook)

    def emit(self, output) -> None:
        for hook in list(self.hooks):
            hook(self, (), output)


def fake_modules() -> FinalCircuitModules:
    return FinalCircuitModules(
        h_relu=FakeModule(),
        predicate_linear=FakeModule(),
        predicate_relu=FakeModule(),
        readout_linear=FakeModule(),
        output_relu=FakeModule(),
    )


class ScopedActivationHookTests(unittest.TestCase):
    def test_captures_each_role_and_removes_every_hook(self) -> None:
        modules = fake_modules()
        hooks = ScopedActivationHooks(modules)

        with hooks:
            hooks.begin_inference()
            for role, module in modules.by_role().items():
                module.emit(f"value-{role}")
            captured = hooks.captured()

        self.assertEqual(
            captured,
            {role: f"value-{role}" for role in modules.by_role()},
        )
        self.assertTrue(hooks.hooks_removed)
        self.assertTrue(all(not module.hooks for module in modules.by_role().values()))

    def test_rejects_duplicate_hook_firings_for_one_inference(self) -> None:
        modules = fake_modules()
        with ScopedActivationHooks(modules) as hooks:
            hooks.begin_inference()
            modules.h_relu.emit("first")
            with self.assertRaisesRegex(ActivationExecutionError, "more than once"):
                modules.h_relu.emit("second")

    def test_rejects_incomplete_capture(self) -> None:
        modules = fake_modules()
        with ScopedActivationHooks(modules) as hooks:
            hooks.begin_inference()
            modules.h_relu.emit("only-one")
            with self.assertRaisesRegex(ActivationExecutionError, "coverage"):
                hooks.captured()


if __name__ == "__main__":
    unittest.main()
