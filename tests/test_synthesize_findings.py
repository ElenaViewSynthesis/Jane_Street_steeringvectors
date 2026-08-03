from __future__ import annotations

import unittest

from scripts.synthesize_findings import (
    render_markdown,
    summarize_semantic_interventions,
)


class SemanticInterventionSummaryTests(unittest.TestCase):
    def test_summarizes_crossings_and_errors(self) -> None:
        activation = {
            "tensors": {"output": {"values": [0.0]}, "readout": {"values": [-3.0]}}
        }
        comparison = {
            "predicate_delta_max_abs_error": 0.0,
            "readout_delta_abs_error": 1e-6,
            "output_delta_abs_error": 0.0,
            "predicate_relu_crossing_rows": [2],
            "final_relu_crossed": False,
        }
        report = {
            "results": [
                {
                    "interventions": [
                        {"observations": [{"activation": activation, "comparison": comparison}]}
                    ]
                }
            ]
        }

        summary = summarize_semantic_interventions(report)

        self.assertEqual(summary["observation_count"], 1)
        self.assertEqual(summary["predicate_relu_crossing_observations"], 1)
        self.assertEqual(summary["readout_delta_max_abs_error"], 1e-6)


class MarkdownRenderingTests(unittest.TestCase):
    def test_renders_core_claims(self) -> None:
        summary = {
            "architecture": {
                "linear_layer_count": 2,
                "relu_layer_count": 2,
                "parameter_count": 10,
                "input_character_limit": 55,
                "target_digest_hex": "abcd",
            },
            "behavioral_probing": {"observation_count": 4},
            "semantic_directions": {
                "left_held_out_accuracy": 0.2,
                "right_held_out_accuracy": 0.2,
                "chance_accuracy": 0.2,
                "numerical_rank": 2,
                "entropy_effective_rank": 1.5,
                "same_slot_category_cross_fold_mean_cosine": 0.1,
                "same_fold_category_cross_slot_mean_cosine": -0.1,
                "cross_fold_model": {
                    "observation_count": 150,
                    "block_count": 6,
                    "same_category_observation_count": 30,
                    "different_category_observation_count": 120,
                    "same_category_mean": 0.25,
                    "different_category_mean": -0.06,
                    "same_minus_different": 0.31,
                    "permutation_repetitions": 10000,
                    "permutation_two_sided_p_value": 0.0001,
                },
            },
            "causal_controls": {
                "canonical": {"observation_count": 3},
                "semantic": {
                    "observation_count": 6,
                    "predicate_delta_max_abs_error": 0.0,
                    "readout_delta_max_abs_error": 1e-6,
                    "predicate_relu_crossing_observations": 2,
                },
            },
            "local_geometry": {
                "predicate_jacobian_shape": [16, 192],
                "predicate_jacobian_nonzero_count": 192,
                "analytic_autograd_jacobian_max_abs_error": 0.0,
                "endpoint_count": 4,
                "endpoints_with_predicate_relu_crossings": 2,
                "crossing_observation_count": 4,
                "crossing_mismatch_count": 0,
                "readout_hessian_frobenius_norm": 0.0,
                "output_hessian_frobenius_norm": 0.0,
            },
            "remaining_uncertainty": ["Unicode remains unresolved."],
        }

        rendered = render_markdown(summary)

        self.assertIn("rigorous null result", rendered)
        self.assertIn("estimated alignment contrast of 0.310", rendered)
        self.assertIn("two-sided p=0.0001", rendered)
        self.assertIn("PyHessian is not used", rendered)
        self.assertIn("Unicode remains unresolved", rendered)


if __name__ == "__main__":
    unittest.main()
