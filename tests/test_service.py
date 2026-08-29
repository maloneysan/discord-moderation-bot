import json
from pathlib import Path
import unittest
from unittest.mock import AsyncMock

from discord_moderation_bot.engine import ModerationEngine
from discord_moderation_bot.service import GroqModerationService, _RetryableApiError


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
        service._post_chat.assert_awaited_once_with(
            "婉曲な表現", "会話の文脈", (), request_source="text"
        )
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
        service._post_chat.assert_not_awaited()

    async def test_api_failure_uses_local_fallback_without_source_logging(self) -> None:
        service = self.make_service()
        service._post_chat = AsyncMock(side_effect=RuntimeError("network"))

        result = await service.analyze("外国人についての婉曲表現")

        self.assertFalse(result.detected)
        self.assertEqual(service._failed_requests, 1)

    async def test_api_failure_uses_relaxed_compositional_local_detection(self) -> None:
        service = self.make_service()
        service._post_chat = AsyncMock(side_effect=RuntimeError("network"))

        result = await service.analyze("お前の話はどうでもいい")

        self.assertEqual({item.category for item in result.detections}, {"cynicism"})

    async def test_quota_gate_uses_relaxed_compositional_local_detection(self) -> None:
        service = GroqModerationService(
            "not-a-real-key",
            ModerationEngine.from_json(RULES_PATH),
            text_interval_seconds=60,
        )
        service._post_chat = AsyncMock(
            return_value={
                "discrimination": {"detected": False, "confidence": 1, "reason": ""},
                "cynicism": {"detected": False, "confidence": 1, "reason": ""},
                "sexual_content": {"detected": False, "confidence": 1, "reason": ""},
                "sensitive_term": {"detected": False, "confidence": 1, "reason": ""},
                "drug_content": {"detected": False, "confidence": 1, "reason": ""},
            }
        )

        await service.analyze("外国人についての一件目")
        result = await service.analyze("お前の話はどうでもいい")

        self.assertEqual({item.category for item in result.detections}, {"cynicism"})

    async def test_token_aware_gate_skips_excess_remote_requests(self) -> None:
        service = GroqModerationService(
            "not-a-real-key",
            ModerationEngine.from_json(RULES_PATH),
            text_interval_seconds=60,
        )
        service._post_chat = AsyncMock(
            return_value={
                "discrimination": {"detected": False, "confidence": 1, "reason": ""},
                "cynicism": {"detected": False, "confidence": 1, "reason": ""},
                "sexual_content": {"detected": False, "confidence": 1, "reason": ""},
                "sensitive_term": {"detected": False, "confidence": 1, "reason": ""},
                "drug_content": {"detected": False, "confidence": 1, "reason": ""},
            }
        )

        await service.analyze("外国人についての一件目")
        await service.analyze("外国人についての二件目")

        self.assertEqual(service._post_chat.await_count, 1)
        self.assertEqual(service._quota_skipped_text_requests, 1)

    async def test_no_internal_daily_cap_for_text_or_voice_ai(self) -> None:
        service = GroqModerationService(
            "not-a-real-key",
            ModerationEngine.from_json(RULES_PATH),
            text_interval_seconds=0,
            voice_analysis_interval_seconds=0,
        )
        service._post_chat = AsyncMock(
            return_value={
                "discrimination": {"detected": False, "confidence": 1, "reason": ""},
                "cynicism": {"detected": False, "confidence": 1, "reason": ""},
                "sexual_content": {"detected": False, "confidence": 1, "reason": ""},
                "sensitive_term": {"detected": False, "confidence": 1, "reason": ""},
                "drug_content": {"detected": False, "confidence": 1, "reason": ""},
            }
        )

        for index in range(71):
            await service.analyze(f"外国人についての確認{index}")
        for index in range(301):
            await service.analyze(
                f"通常のVC文字起こし{index}", request_source="voice"
            )

        self.assertEqual(service._post_chat.await_count, 372)

    async def test_ordinary_message_does_not_spend_remote_quota(self) -> None:
        service = self.make_service()
        service._post_chat = AsyncMock()

        result = await service.analyze("今日はいい天気ですね")

        self.assertFalse(result.detected)
        service._post_chat.assert_not_awaited()

    async def test_primary_429_uses_secondary_text_model(self) -> None:
        service = self.make_service()
        category_payload = {
            "discrimination": {"detected": True, "confidence": 90, "reason": "排除です。"},
            "cynicism": {"detected": False, "confidence": 1, "reason": ""},
            "sexual_content": {"detected": False, "confidence": 1, "reason": ""},
            "sensitive_term": {"detected": False, "confidence": 1, "reason": ""},
            "drug_content": {"detected": False, "confidence": 1, "reason": ""},
        }
        service._request_json = AsyncMock(
            side_effect=[
                _RetryableApiError("daily tokens", status=429, retry_after_seconds=120),
                {
                    "choices": [
                        {"message": {"content": json.dumps(category_payload)}}
                    ]
                },
            ]
        )

        payload = await service._post_chat("外国人について", None, ())

        self.assertEqual(payload, category_payload)
        calls = service._request_json.await_args_list
        self.assertEqual(calls[0].kwargs["json_body"]["model"], "openai/gpt-oss-120b")
        self.assertEqual(calls[1].kwargs["json_body"]["model"], "openai/gpt-oss-20b")
        self.assertEqual(service._fallback_model_requests, 1)

    async def test_voice_uses_dedicated_safeguard_model_and_compact_policy(self) -> None:
        service = self.make_service()
        service._request_json = AsyncMock(
            return_value={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "discrimination": {
                                        "detected": False,
                                        "confidence": 1,
                                        "reason": "",
                                    },
                                    "cynicism": {
                                        "detected": False,
                                        "confidence": 1,
                                        "reason": "",
                                    },
                                    "sexual_content": {
                                        "detected": False,
                                        "confidence": 1,
                                        "reason": "",
                                    },
                                    "sensitive_term": {
                                        "detected": False,
                                        "confidence": 1,
                                        "reason": "",
                                    },
                                    "drug_content": {
                                        "detected": False,
                                        "confidence": 1,
                                        "reason": "",
                                    },
                                }
                            )
                        }
                    }
                ]
            }
        )

        await service._post_chat(
            "音声文字起こし", None, (), request_source="voice"
        )

        body = service._request_json.await_args.kwargs["json_body"]
        self.assertEqual(body["model"], "openai/gpt-oss-safeguard-20b")
        self.assertEqual(
            body["messages"][0]["content"], service._VOICE_SYSTEM_PROMPT
        )

    async def test_audio_api_failure_uses_local_speech_fallback(self) -> None:
        fallback = AsyncMock()
        fallback.transcribe_wav.return_value = "ローカル認識結果"
        service = GroqModerationService(
            "not-a-real-key",
            ModerationEngine.from_json(RULES_PATH),
            local_speech_transcriber=fallback,
        )
        service._post_audio = AsyncMock(side_effect=RuntimeError("audio-api"))

        transcript = await service.transcribe_wav(b"RIFF-test")

        self.assertEqual(transcript, "ローカル認識結果")
        fallback.transcribe_wav.assert_awaited_once_with(b"RIFF-test")
        self.assertEqual(service._local_audio_fallbacks, 1)

    async def test_audio_gate_falls_back_instead_of_building_a_queue(self) -> None:
        fallback = AsyncMock()
        fallback.transcribe_wav.return_value = "二件目のローカル認識"
        service = GroqModerationService(
            "not-a-real-key",
            ModerationEngine.from_json(RULES_PATH),
            audio_interval_seconds=60,
            local_speech_transcriber=fallback,
        )
        service._post_audio = AsyncMock(return_value={"text": "一件目の外部認識"})

        first = await service.transcribe_wav(b"RIFF-one")
        second = await service.transcribe_wav(b"RIFF-two")

        self.assertEqual(first, "一件目の外部認識")
        self.assertEqual(second, "二件目のローカル認識")
        self.assertEqual(service._post_audio.await_count, 1)
        self.assertEqual(service._quota_skipped_audio_requests, 1)

    async def test_flagged_local_speech_fallback_is_also_verified(self) -> None:
        fallback = AsyncMock()
        fallback.transcribe_wav.return_value = "うお"
        service = GroqModerationService(
            "not-a-real-key",
            ModerationEngine.from_json(RULES_PATH),
            audio_interval_seconds=60,
            local_speech_transcriber=fallback,
        )
        service._post_audio = AsyncMock(
            side_effect=[{"text": "通常の会話"}, {"text": "一件目"}]
        )

        await service.transcribe_wav(b"RIFF-one")
        transcript = await service.transcribe_wav(b"RIFF-two")

        self.assertEqual(transcript, "")
        self.assertEqual(service._unverified_audio_segments, 1)

    async def test_compositional_vosk_hit_survives_verifier_api_limit(self) -> None:
        fallback = AsyncMock()
        fallback.transcribe_wav.return_value = "外国人は出ていけ"
        service = GroqModerationService(
            "not-a-real-key",
            ModerationEngine.from_json(RULES_PATH),
            audio_interval_seconds=60,
            local_speech_transcriber=fallback,
        )
        service._next_audio_request_at = float("inf")
        service._post_audio = AsyncMock(side_effect=RuntimeError("quota"))

        transcript = await service.transcribe_wav(b"RIFF-test")

        self.assertEqual(transcript, "外国人は出ていけ")
        self.assertEqual(service._offline_verified_audio_segments, 1)

    async def test_short_vosk_hit_is_suppressed_without_external_verification(self) -> None:
        fallback = AsyncMock()
        fallback.transcribe_wav.return_value = "うお"
        service = GroqModerationService(
            "not-a-real-key",
            ModerationEngine.from_json(RULES_PATH),
            audio_interval_seconds=60,
            local_speech_transcriber=fallback,
        )
        service._next_audio_request_at = float("inf")
        service._post_audio = AsyncMock(side_effect=RuntimeError("quota"))

        transcript = await service.transcribe_wav(b"RIFF-test")

        self.assertEqual(transcript, "")

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

    async def test_flagged_voice_transcript_requires_independent_agreement(self) -> None:
        service = self.make_service()
        service._post_audio = AsyncMock(
            side_effect=[{"text": "うお"}, {"text": "今日は"}]
        )

        transcript = await service.transcribe_wav(b"RIFF-test")

        self.assertEqual(transcript, "")
        self.assertEqual(service._unverified_audio_segments, 1)
        self.assertEqual(service._post_audio.await_count, 2)

    async def test_flagged_voice_transcript_is_kept_after_exact_agreement(self) -> None:
        service = self.make_service()
        service._post_audio = AsyncMock(
            side_effect=[{"text": "うお。"}, {"text": "うお"}]
        )

        transcript = await service.transcribe_wav(b"RIFF-test")

        self.assertEqual(transcript, "うお。")
        self.assertEqual(service._post_audio.await_count, 2)

    async def test_voice_text_without_local_rule_reaches_contextual_ai(self) -> None:
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

        await service.analyze("今日はいい天気ですね", request_source="voice")

        service._post_chat.assert_awaited_once()

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

        await service.analyze(
            "外国人について", recent_context=("一", "二", "三", "四")
        )

        service._post_chat.assert_awaited_once_with(
            "外国人について",
            None,
            ("一", "二", "三", "四"),
            request_source="text",
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

    def test_speech_prompt_does_not_bias_recognition_toward_watched_terms(self) -> None:
        prompt = GroqModerationService._SPEECH_PROMPT
        self.assertIn("実際に聞こえた内容だけ", prompt)
        self.assertIn("推測で語を補わず", prompt)
        for phrase in ("うお", "どわー", "クイヤ", "めう", "ADHD"):
            self.assertNotIn(phrase, prompt)


if __name__ == "__main__":
    unittest.main()
