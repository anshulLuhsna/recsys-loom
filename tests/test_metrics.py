import unittest

from recsys_loom.metrics import (
    average_precision_at_k,
    ranking_metrics_at_k,
    recall_at_k,
    retrieval_metrics_at_k,
)


class AveragePrecisionAtKTests(unittest.TestCase):
    def test_perfect_ranking_scores_one(self) -> None:
        self.assertEqual(
            average_precision_at_k(
                relevant={"article-a", "article-b"},
                predicted=["article-a", "article-b"],
                k=12,
            ),
            1.0,
        )

    def test_no_relevant_prediction_scores_zero(self) -> None:
        self.assertEqual(
            average_precision_at_k(
                relevant={"article-a"},
                predicted=["article-b", "article-c"],
                k=12,
            ),
            0.0,
        )

    def test_single_hit_at_rank_three_scores_one_third(self) -> None:
        self.assertAlmostEqual(
            average_precision_at_k(
                relevant={"article-c"},
                predicted=["article-a", "article-b", "article-c"],
                k=12,
            ),
            1.0 / 3.0,
        )

    def test_duplicate_prediction_cannot_earn_credit_twice(self) -> None:
        self.assertEqual(
            average_precision_at_k(
                relevant={"article-a"},
                predicted=["article-a", "article-a"],
                k=12,
            ),
            1.0,
        )

    def test_predictions_after_rank_twelve_are_ignored(self) -> None:
        self.assertEqual(
            average_precision_at_k(
                relevant={"article-hit"},
                predicted=[f"miss-{index}" for index in range(12)] + ["article-hit"],
                k=12,
            ),
            0.0,
        )

    def test_twelve_perfect_hits_score_one_when_more_items_are_relevant(self) -> None:
        relevant = {f"article-{index}" for index in range(13)}
        predictions = [f"article-{index}" for index in range(12)]
        self.assertEqual(average_precision_at_k(relevant, predictions, k=12), 1.0)


class RankingMetricsAtKTests(unittest.TestCase):
    def test_aggregate_metrics_match_worked_two_customer_example(self) -> None:
        metrics = ranking_metrics_at_k(
            relevant_by_customer={
                "customer-1": {"article-a"},
                "customer-2": {"article-c", "article-d"},
            },
            predictions_by_customer={
                "customer-1": ["article-a", "article-x"],
                "customer-2": ["article-x", "article-c"],
            },
            catalog_size=4,
            k=12,
        )

        self.assertEqual(metrics["customers"], 2)
        self.assertEqual(metrics["customers_with_hit"], 2)
        self.assertEqual(metrics["matched_relevance_pairs"], 2)
        self.assertAlmostEqual(metrics["map_at_12"], 0.625)
        self.assertAlmostEqual(metrics["recall_at_12"], 0.75)
        self.assertAlmostEqual(metrics["micro_recall_at_12"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["hit_rate_at_12"], 1.0)
        self.assertAlmostEqual(metrics["catalog_coverage_at_12"], 0.75)


class RecallAtKTests(unittest.TestCase):
    def test_all_relevant_found_scores_one(self) -> None:
        self.assertEqual(
            recall_at_k({"a", "b"}, ["a", "b", "c", "d"], k=4),
            1.0,
        )

    def test_none_found_scores_zero(self) -> None:
        self.assertEqual(
            recall_at_k({"a"}, ["b", "c", "d"], k=3),
            0.0,
        )

    def test_partial_recovery(self) -> None:
        self.assertAlmostEqual(
            recall_at_k({"a", "b", "c"}, ["a", "x", "b", "y"], k=4),
            2.0 / 3.0,
        )

    def test_items_beyond_k_are_ignored(self) -> None:
        self.assertEqual(
            recall_at_k({"z"}, ["a", "b", "z"], k=2),
            0.0,
        )

    def test_empty_relevant_returns_zero(self) -> None:
        self.assertEqual(recall_at_k(set(), ["a", "b"], k=2), 0.0)

    def test_order_does_not_matter_within_k(self) -> None:
        relevant = {"a", "c"}
        self.assertEqual(
            recall_at_k(relevant, ["c", "b", "a"], k=3),
            recall_at_k(relevant, ["a", "b", "c"], k=3),
        )


class RetrievalMetricsAtKTests(unittest.TestCase):
    def test_two_customer_example_at_multiple_cutoffs(self) -> None:
        relevant = {
            "c1": {"a", "b"},
            "c2": {"c"},
        }
        candidates = {
            "c1": ["x", "a", "y", "b"],
            "c2": ["c", "x", "y"],
        }
        metrics = retrieval_metrics_at_k(
            relevant, candidates, catalog_size=5, k_values=(2, 4),
        )

        self.assertEqual(metrics["customers"], 2)
        self.assertEqual(metrics["relevance_pairs"], 3)

        self.assertAlmostEqual(metrics["recall_at_2"], (0.5 + 1.0) / 2)
        self.assertAlmostEqual(metrics["recall_at_4"], (1.0 + 1.0) / 2)

        self.assertEqual(metrics["customers_with_hit_at_2"], 2)
        self.assertEqual(metrics["matched_pairs_at_2"], 2)
        self.assertEqual(metrics["matched_pairs_at_4"], 3)

    def test_customer_with_no_candidates_scores_zero(self) -> None:
        metrics = retrieval_metrics_at_k(
            {"c1": {"a"}}, {}, catalog_size=1, k_values=(5,),
        )
        self.assertAlmostEqual(metrics["recall_at_5"], 0.0)
        self.assertEqual(metrics["customers_with_hit_at_5"], 0)


if __name__ == "__main__":
    unittest.main()
