from __future__ import annotations

import copy
import unittest

from scripts.activation_schema import PREDICATE_TARGETS, decode_q_values, float32_sha256, predicate_values_from_q
from scripts.intervention_schema import canonical_target_h192, validate_intervention_spec
from scripts.probe_schema import ProbeSchemaError


class InterventionSchemaTests(unittest.TestCase):
    def test_canonical_target_satisfies_every_recovered_predicate(self) -> None:
        target = canonical_target_h192()

        self.assertEqual(len(target), 192)
        self.assertEqual(predicate_values_from_q(decode_q_values(target)), PREDICATE_TARGETS)

    def test_spec_binds_vector_hash_and_manifest_case_ids(self) -> None:
        vector = canonical_target_h192()
        spec = {
            "schema_version": 1,
            "suite_id": "test-suite",
            "description": "Test intervention.",
            "source_activation_report_sha256": "a" * 64,
            "source_manifest_sha256": "b" * 64,
            "interventions": [
                {
                    "id": "target",
                    "mode": "replace",
                    "vector": vector,
                    "vector_float32_sha256": float32_sha256(vector),
                    "strengths": [1.0],
                    "case_ids": ["case-a"],
                    "source": "unit test",
                }
            ],
        }
        self.assertIs(validate_intervention_spec(spec, case_ids={"case-a"}), spec)

        tampered = copy.deepcopy(spec)
        tampered["interventions"][0]["vector"][0] = 1.0 - tampered["interventions"][0]["vector"][0]
        with self.assertRaisesRegex(ProbeSchemaError, "hash"):
            validate_intervention_spec(tampered, case_ids={"case-a"})

        unknown_case = copy.deepcopy(spec)
        unknown_case["interventions"][0]["case_ids"] = ["missing"]
        with self.assertRaisesRegex(ProbeSchemaError, "case_ids"):
            validate_intervention_spec(unknown_case, case_ids={"case-a"})


if __name__ == "__main__":
    unittest.main()
