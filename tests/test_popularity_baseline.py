import tempfile
import unittest
from pathlib import Path

from recsys_loom.popularity import run_popularity_baseline


class PopularityBaselineTests(unittest.TestCase):
    def test_training_counts_rank_articles_and_score_hidden_week(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            transactions_path = root / "transactions.csv"
            articles_path = root / "articles.csv"
            output_directory = root / "output"

            transactions_path.write_text(
                "t_dat,customer_id,article_id,price,sales_channel_id\n"
                "2020-09-14,customer-1,article-a,0.1,1\n"
                "2020-09-15,customer-2,article-a,0.1,1\n"
                "2020-09-15,customer-2,article-b,0.1,1\n"
                "2020-09-16,customer-1,article-a,0.1,1\n"
                "2020-09-16,customer-3,article-b,0.1,1\n"
                "2020-09-16,customer-4,article-c,0.1,1\n",
                encoding="utf-8",
            )
            articles_path.write_text(
                "article_id\narticle-a\narticle-b\narticle-c\n",
                encoding="utf-8",
            )

            result = run_popularity_baseline(
                transactions_path=transactions_path,
                articles_path=articles_path,
                output_directory=output_directory,
                train_end="2020-09-15",
                validation_start="2020-09-16",
                validation_end="2020-09-22",
                k=2,
                threads=1,
            )

            self.assertEqual(
                result["top_articles"],
                [
                    {"article_id": "article-a", "purchase_count": 2},
                    {"article_id": "article-b", "purchase_count": 1},
                ],
            )
            self.assertAlmostEqual(result["aggregate"]["map_at_2"], 0.5)
            self.assertAlmostEqual(result["aggregate"]["recall_at_2"], 2.0 / 3.0)
            self.assertAlmostEqual(result["aggregate"]["hit_rate_at_2"], 2.0 / 3.0)
            self.assertAlmostEqual(result["aggregate"]["catalog_coverage_at_2"], 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
