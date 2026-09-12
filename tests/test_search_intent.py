import unittest

from recsys_loom.search.intent import parse_query
from recsys_loom.search.pipeline import _rrf


class ParseQueryTests(unittest.TestCase):
    def test_hoodie_query_sets_hard_type_and_color(self) -> None:
        intent = parse_query("black oversized hoodie for men")
        self.assertEqual(intent.color, "Black")
        self.assertEqual(intent.product_type, "Hoodie")
        self.assertEqual(intent.section, "Menswear")
        self.assertIn("oversized", intent.style_terms)
        self.assertEqual(intent.hard_constraints["product_type"], "Hoodie")
        self.assertEqual(intent.hard_constraints["color"], "Black")

    def test_style_only_query_stays_soft(self) -> None:
        intent = parse_query("minimal summer linen")
        self.assertIsNone(intent.product_type)
        self.assertEqual(set(intent.style_terms), {"minimal", "summer", "linen"})
        self.assertEqual(intent.hard_constraints, {})

    def test_color_without_type_is_soft(self) -> None:
        intent = parse_query("red")
        self.assertEqual(intent.color, "Red")
        self.assertNotIn("color", intent.hard_constraints)
        self.assertEqual(intent.soft_preferences["color"], "Red")


class FusionTests(unittest.TestCase):
    def test_reciprocal_rank_prefers_early_ranks(self) -> None:
        self.assertGreater(_rrf(1), _rrf(10))
