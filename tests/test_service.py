from pathlib import Path
import unittest
from unittest.mock import AsyncMock

from discord_moderation_bot.engine import ModerationEngine
from discord_moderation_bot.service import GroqModerationService


RULES_PATH = Path(__file__).resolve().parent.parent / "config" / "rules.json"


class GroqModerationServiceTests(unittest.IsolatedAsyncioTestCase):
    def make_service(
        self, threshold: int = 50, cynicism_threshold: int = 80
    ) -> GroqModerationService:
        return GroqModerationService(
            "not-a-real-key",
            ModerationEngine.from_json(RULES_PATH),
            confidence_threshold=threshold,
            cynicism_confidence_threshold=cynicism_threshold,
        )

    async def test_contextual_discrimination_and_cynicism_are_combined(self) -> None:
        service = self.make_service()
        service._post_chat = AsyncMock(
            return_value={
                "discrimination": {
                    "detected": True,
                    "confidence": 94,
                    "reason": "属性を理由に排除する内容です。",
                },
                "cynicism": {
                    "detected": True,
                    "confidence": 83,
                    "reason": "相手を嘲笑する内容です。",
                },
                "sexual_content": {"detected": False, "confidence": 1, "reason": ""},
                "sensitive_term": {"detected": False, "confidence": 1, "reason": ""},
                "drug_content": {"detected": False, "confidence": 1, "reason": ""},
            }
        )

        result = await service.analyze("婉曲な表現", reply_context="会話の文脈")

        self.assertEqual(
            {item.category for item in result.detections},
            {"discrimination", "cynicism"},
        )
        service._post_chat.assert_awaited_once_with("婉曲な表現", "会話の文脈", ())
        self.assertNotIn("婉曲な表現", repr(result))
        self.assertIn("排除", result.detections[0].reason)

    async def test_local_rules_remain_a_safety_net_for_community_terms(self) -> None:
        service = self.make_service()
        service._post_chat = AsyncMock(
            return_value={
                "discrimination": {
                    "detected": False,
                    "confidence": 3,
                    "reason": "",
                },
                "cynicism": {
                    "detected": False,
                    "confidence": 10,
                    "reason": "",
                },
                "sexual_content": {"detected": False, "confidence": 1, "reason": ""},
                "sensitive_term": {"detected": False, "confidence": 1, "reason": ""},
                "drug_content": {"detected": False, "confidence": 1, "reason": ""},
            }
        )

        result = await service.analyze("うお")

        self.assertEqual({item.category for item in result.detections}, {"cynicism"})

    async def test_api_failure_uses_local_fallback_without_source_logging(self) -> None:
        service = self.make_service()
        service._post_chat = AsyncMock(side_effect=RuntimeError("network"))

        result = await service.analyze("外国人は出ていけ")

        self.assertTrue(result.detected)
        self.assertEqual(service._failed_requests, 1)

    async def test_category_specific_cynicism_threshold_is_enforced(self) -> None:
        service = self.make_service(threshold=60)
        result = service._result_from_payload(
            {
                "discrimination": {
                    "detected": True,
                    "confidence": 59,
                    "reason": "属性への敵意です。",
                },
                "cynicism": {
                    "detected": True,
                    "confidence": 79,
                    "reason": "相手への嘲笑です。",
                },
                "sexual_content": {"detected": False, "confidence": 1, "reason": ""},
                "sensitive_term": {"detected": False, "confidence": 1, "reason": ""},
                "drug_content": {"detected": False, "confidence": 1, "reason": ""},
            }
        )
        self.assertFalse(result.detected)

        result = service._result_from_payload(
            {
                "discrimination": {"detected": False, "confidence": 1, "reason": ""},
                "cynicism": {
                    "detected": True,
                    "confidence": 80,
                    "reason": "相手への明確な嘲笑です。",
                },
                "sexual_content": {"detected": False, "confidence": 1, "reason": ""},
                "sensitive_term": {"detected": False, "confidence": 1, "reason": ""},
                "drug_content": {"detected": False, "confidence": 1, "reason": ""},
            }
        )
        self.assertEqual({item.category for item in result.detections}, {"cynicism"})

    def test_global_threshold_can_be_stricter_than_cynicism_threshold(self) -> None:
        service = self.make_service(threshold=90, cynicism_threshold=80)
        result = service._result_from_payload(
            {
                "discrimination": {"detected": False, "confidence": 1, "reason": ""},
                "cynicism": {
                    "detected": True,
                    "confidence": 89,
                    "reason": "相手への嘲笑です。",
                },
                "sexual_content": {"detected": False, "confidence": 1, "reason": ""},
                "sensitive_term": {"detected": False, "confidence": 1, "reason": ""},
                "drug_content": {"detected": False, "confidence": 1, "reason": ""},
            }
        )

        self.assertFalse(result.detected)

    async def test_audio_transcript_is_returned_but_not_retained(self) -> None:
        service = self.make_service()
        service._post_audio = AsyncMock(return_value={"text": "音声の認識結果"})

        transcript = await service.transcribe_wav(b"RIFF-test")

        self.assertEqual(transcript, "音声の認識結果")
        self.assertNotIn("音声の認識結果", repr(service.__dict__))

    async def test_recent_messages_are_forwarded_as_bounded_context(self) -> None:
        service = self.make_service()
        service._post_chat = AsyncMock(
            return_value={
                "discrimination": {"detected": False, "confidence": 1, "reason": ""},
                "cynicism": {"detected": False, "confidence": 1, "reason": ""},
                "sexual_content": {"detected": False, "confidence": 1, "reason": ""},
                "sensitive_term": {"detected": False, "confidence": 1, "reason": ""},
                "drug_content": {"detected": False, "confidence": 1, "reason": ""},
            }
        )

        await service.analyze("現在", recent_context=("一", "二", "三", "四"))

        service._post_chat.assert_awaited_once_with(
            "現在", None, ("一", "二", "三", "四")
        )

    def test_low_confidence_and_no_speech_segments_are_removed(self) -> None:
        payload = {
            "text": "fallback text",
            "segments": [
                {"text": "信頼できる部分", "avg_logprob": -0.1, "no_speech_prob": 0.1},
                {"text": "短い珍しい語", "avg_logprob": -0.8, "no_speech_prob": 0.1},
                {"text": "低信頼", "avg_logprob": -1.2, "no_speech_prob": 0.1},
                {"text": "無音推定", "avg_logprob": -0.1, "no_speech_prob": 0.9},
            ],
        }
        service = self.make_service()
        self.assertEqual(
            service._trusted_transcript(payload),
            "信頼できる部分 短い珍しい語",
        )

    async def test_prompt_covers_multilingual_coded_and_community_language(self) -> None:
        prompt = GroqModerationService._SYSTEM_PROMPT
        for phrase in (
            "every language",
            "Japanese",
            "dog whistles",
            "うお",
            "どわー",
            "女々しい",
            "クイヤ",
            "めう",
            "SEXUAL_CONTENT",
            "SENSITIVE_TERM",
            "ADHD",
            "DRUG_CONTENT",
            "overdose",
            "飛べるやつ",
            "OD缶",
            "without quoting",
            "recent_messages",
            "identifiable person or group target",
            "friendly banter",
        ):
            self.assertIn(phrase, prompt)

    def test_speech_prompt_preserves_short_moderation_vocabulary(self) -> None:
        prompt = GroqModerationService._SPEECH_PROMPT
        for phrase in ("短い発言", "俗語", "薬物名", "うお", "ADHD"):
            self.assertIn(phrase, prompt)


if __name__ == "__main__":
    unittest.main()
