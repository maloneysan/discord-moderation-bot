from pathlib import Path
import unittest

from discord_moderation_bot.evaluation import evaluate_cases, load_evaluation_cases
from discord_moderation_bot.models import CategoryDetection, DetectionResult


DATASET = Path(__file__).resolve().parent.parent / "config" / "evaluation_cases.json"


class _FakeService:
    async def analyze(self, text, *, reply_context=None, recent_context=()):
        detected = ()
        if text == "positive":
            detected = (
                CategoryDetection("cynicism", "冷笑", 90, ("test",)),
            )
        return DetectionResult(bool(detected), detected)


class EvaluationDatasetTests(unittest.IsolatedAsyncioTestCase):
    def test_dataset_covers_required_dimensions_without_duplicate_ids(self) -> None:
        cases = load_evaluation_cases(DATASET)
        dimensions = {case.dimension for case in cases}
        self.assertGreaterEqual(len(cases), 25)
        self.assertIn("複数投稿文脈", dimensions)
        self.assertIn("引用と批判", dimensions)
        self.assertIn("伏字・難読化", dimensions)
        self.assertEqual(len({case.case_id for case in cases}), len(cases))

    async def test_report_calculates_precision_recall_and_mismatch_ids(self) -> None:
        cases = load_evaluation_cases(DATASET)[:2]
        cases = (
            cases[0].__class__("one", "test", "positive", frozenset({"cynicism"})),
            cases[0].__class__("two", "test", "positive", frozenset()),
            cases[0].__class__("three", "test", "negative", frozenset({"cynicism"})),
        )
        report = await evaluate_cases(_FakeService(), cases)

        self.assertEqual(report.exact_matches, 1)
        self.assertEqual(report.true_positives, 1)
        self.assertEqual(report.false_positives, 1)
        self.assertEqual(report.false_negatives, 1)
        self.assertEqual(report.mismatched_case_ids, ("two", "three"))
        self.assertAlmostEqual(report.precision, 0.5)
        self.assertAlmostEqual(report.recall, 0.5)


if __name__ == "__main__":
    unittest.main()
