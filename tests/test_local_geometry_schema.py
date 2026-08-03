from __future__ import annotations

import copy
import math
import unittest

from scripts.local_geometry_schema import (
    analytic_predicate_jacobian,
    cross_fold_cosine_rows,
    direction_records,
    gram_matrix,
    predicate_boundaries,
    readout_gradient,
    validate_semantic_analysis,
)
from scripts.probe_schema import ProbeSchemaError


def semantic_analysis_fixture() -> dict:
    categories = ["animal", "color", "food", "place", "verb"]
    slots = {}
    for slot in ("left", "right"):
        folds = []
        for fold_index in range(3):
            directions = {}
            predictions = []
            for category_index, category in enumerate(categories):
                vector = [0.0] * 192
                vector[(fold_index * 10 + category_index) % 192] = 1.0
                directions[category] = vector
                predictions.extend(
                    {
                        "case_id": f"{slot}-{fold_index}-{category}-{case_index}",
                        "actual": category,
                        "predicted": category,
                        "correct": True,
                    }
                    for case_index in range(10)
                )
            folds.append(
                {
                    "fold_index": fold_index,
                    "directions": directions,
                    "evaluation": {"predictions": predictions},
                }
            )
        slots[slot] = {"folds": folds}
    return {
        "schema_version": 1,
        "analysis_kind": "held_out_semantic_directions",
        "source": {
            "activation_report_sha256": "a" * 64,
            "artifact_sha256": "b" * 64,
            "manifest_sha256": "c" * 64,
            "suite_id": "semantic-test",
        },
        "protocol": {},
        "categories": categories,
        "words_by_category": {},
        "slots": slots,
        "interpretation": "fixture",
    }


class PredicateJacobianTests(unittest.TestCase):
    def test_exact_block_coefficients_and_sparsity(self) -> None:
        jacobian = analytic_predicate_jacobian()

        self.assertEqual((len(jacobian), len(jacobian[0])), (16, 192))
        self.assertEqual(sum(value != 0.0 for row in jacobian for value in row), 192)
        self.assertEqual(jacobian[0][:8], [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])
        self.assertEqual(jacobian[4][32:40], jacobian[0][:8])
        self.assertEqual(jacobian[4][64:72], [-2.0 * value for value in jacobian[0][:8]])

    def test_known_gram_matrix(self) -> None:
        self.assertEqual(gram_matrix([[1.0, 0.0], [1.0, 1.0]]), [[1.0, 1.0], [1.0, 2.0]])


class CrossFoldCosineDesignTests(unittest.TestCase):
    def test_builds_six_complete_blocked_comparisons(self) -> None:
        records = direction_records(semantic_analysis_fixture())
        vectors = [record["vector"] for record in records]

        rows = cross_fold_cosine_rows(records, gram_matrix(vectors))

        self.assertEqual(len(rows), 150)
        self.assertEqual(len({row["block_id"] for row in rows}), 6)
        self.assertEqual(sum(row["same_category"] for row in rows), 30)
        self.assertTrue(all(sum(item["block_id"] == block for item in rows) == 25
                            for block in {row["block_id"] for row in rows}))

    def test_rejects_incomplete_direction_grid(self) -> None:
        records = direction_records(semantic_analysis_fixture())[:-1]
        with self.assertRaisesRegex(ProbeSchemaError, "thirty"):
            cross_fold_cosine_rows(records, [[0.0] * 30 for _ in range(30)])


class BoundaryGeometryTests(unittest.TestCase):
    def test_target_bit_layout_has_zero_target_distances(self) -> None:
        h192 = [0.0] * 192
        targets_by_block = [199, 239, 101, 35, 60, 64, 170, 50, 0, 0, 0, 0,
                            194, 185, 172, 227, 117, 149, 250, 124, 0, 0, 0, 0]
        for block, value in enumerate(targets_by_block):
            for bit in range(8):
                h192[8 * block + bit] = float((value >> bit) & 1)

        boundaries = predicate_boundaries(h192)

        self.assertEqual([item["row"] for item in boundaries if item["on_boundary"]], list(range(16, 32)))
        jacobian = analytic_predicate_jacobian()
        expected_gradient = [sum(row[index] for row in jacobian) for index in range(192)]
        self.assertEqual(readout_gradient(h192), expected_gradient)

    def test_zero_state_distance_uses_hyperplane_normal(self) -> None:
        first = predicate_boundaries([0.0] * 192)[0]
        normal_norm = math.sqrt(sum(float(4**bit) for bit in range(8)))
        self.assertAlmostEqual(first["euclidean_distance"], 200.0 / normal_norm)


class SemanticAnalysisValidationTests(unittest.TestCase):
    def test_accepts_complete_thirty_direction_fixture(self) -> None:
        fixture = semantic_analysis_fixture()
        self.assertIs(validate_semantic_analysis(fixture), fixture)

    def test_rejects_non_normalized_direction(self) -> None:
        fixture = copy.deepcopy(semantic_analysis_fixture())
        fixture["slots"]["left"]["folds"][0]["directions"]["animal"][0] = 2.0
        with self.assertRaisesRegex(ProbeSchemaError, "normalized"):
            validate_semantic_analysis(fixture)


if __name__ == "__main__":
    unittest.main()
