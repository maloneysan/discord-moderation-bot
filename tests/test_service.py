from pathlib import Path
import asyncio
import json
import unittest
from unittest.mock import AsyncMock

from discord_moderation_bot.engine import ModerationEngine
from discord_moderation_bot.service import (
    GroqModerationService,
    _RetryableApiError,
    _retry_after_seconds,
)


RULES_PATH = Path(__file__).resolve().parent.parent / "config" / "rules.json"


class GroqModerationServiceTests(unittest.IsolatedAsyncioTestCase):
    def make_service(self, threshold: int = 25) -> GroqModerationService:
        return GroqModerationService(
            "not-a-real-key",
            ModerationEngine.from_json(RULES_PATH),
            confidence_threshold=threshold,
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

        result = await service.analyze("これは文脈依存の確認文です")

        self.assertFalse(result.detected)
        self.assertEqual(service._failed_requests, 1)

    async def test_requested_terms_survive_text_api_failure(self) -> None:
        cases = {
            "うお": {"cynicism"},
            "下ネタ": {"sexual_content"},
            "大麻": {"drug_content"},
            "お前は女々しい": {"discrimination", "cynicism"},
            "了解めう": {"cynicism"},
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                service = self.make_service()
                service._post_chat = AsyncMock(side_effect=RuntimeError("offline"))

                result = await service.analyze(text)

                self.assertEqual(
                    {item.category for item in result.detections}, expected
                )

    async def test_confidence_threshold_is_enforced(self) -> None:
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
                    "confidence": 60,
                    "reason": "相手への嘲笑です。",
                },
                "sexual_content": {"detected": False, "confidence": 1, "reason": ""},
                "sensitive_term": {"detected": False, "confidence": 1, "reason": ""},
                "drug_content": {"detected": False, "confidence": 1, "reason": ""},
            }
        )
        self.assertEqual({item.category for item in result.detections}, {"cynicism"})

    def test_default_threshold_accepts_25_but_not_24(self) -> None:
        service = self.make_service()
        result = service._result_from_payload(
            {
                "discrimination": {
                    "detected": True,
                    "confidence": 24,
                    "reason": "属性への敵意です。",
                },
                "cynicism": {
                    "detected": True,
                    "confidence": 25,
                    "reason": "相手への嘲笑です。",
                },
                "sexual_content": {"detected": False, "confidence": 1, "reason": ""},
                "sensitive_term": {"detected": False, "confidence": 1, "reason": ""},
                "drug_content": {"detected": False, "confidence": 1, "reason": ""},
            }
        )
        self.assertEqual({item.category for item in result.detections}, {"cynicism"})

    async def test_audio_transcript_is_returned_but_not_retained(self) -> None:
        service = self.make_service()
        service._post_audio = AsyncMock(return_value={"text": "音声の認識結果"})

        transcript = await service.transcribe_wav(b"RIFF-test")

        self.assertEqual(transcript, "音声の認識結果")
        self.assertNotIn("音声の認識結果", repr(service.__dict__))

    async def test_audio_api_failure_uses_local_speech_fallback(self) -> None:
        fallback = AsyncMock()
        fallback.transcribe_wav.return_value = "うお"
        service = GroqModerationService(
            "not-a-real-key",
            ModerationEngine.from_json(RULES_PATH),
            local_speech_transcriber=fallback,
        )
        service._post_audio = AsyncMock(
            side_effect=_RetryableApiError(
                "daily limit", status=429, retry_after_seconds=86400
            )
        )

        transcript = await service.transcribe_wav(b"RIFF-test")

        self.assertEqual(transcript, "うお")
        fallback.transcribe_wav.assert_awaited_once_with(b"RIFF-test")
        self.assertEqual(service._local_audio_fallbacks, 1)
        self.assertGreater(service._audio_cooldown_until, 0)

    async def test_turbo_audio_quota_uses_large_v3_fallback(self) -> None:
        service = self.make_service()
        service._request_json = AsyncMock(
            side_effect=[
                _RetryableApiError(
                    "turbo daily limit",
                    status=429,
                    retry_after_seconds=86400,
                ),
                {"text": "うお"},
            ]
        )

        payload = await service._post_audio(b"RIFF-test")

        self.assertEqual(payload["text"], "うお")
        first_form = service._request_json.await_args_list[0].kwargs["form_data"]
        second_form = service._request_json.await_args_list[1].kwargs["form_data"]
        self.assertIn("whisper-large-v3-turbo", repr(first_form._fields))
        self.assertIn("whisper-large-v3", repr(second_form._fields))
        self.assertEqual(service._fallback_audio_model_requests, 1)
        self.assertGreater(service._primary_audio_cooldown_until, 0)

    async def test_audio_cooldown_skips_api_and_uses_local_speech(self) -> None:
        fallback = AsyncMock()
        fallback.transcribe_wav.return_value = "今日は行くめう"
        service = GroqModerationService(
            "not-a-real-key",
            ModerationEngine.from_json(RULES_PATH),
            local_speech_transcriber=fallback,
        )
        service._audio_cooldown_until = float("inf")
        service._post_audio = AsyncMock()

        transcript = await service.transcribe_wav(b"RIFF-test")

        self.assertEqual(transcript, "今日は行くめう")
        service._post_audio.assert_not_awaited()
        self.assertEqual(service._quota_skipped_audio_requests, 1)

    def test_retry_after_parses_daily_request_reset(self) -> None:
        self.assertEqual(
            _retry_after_seconds({"x-ratelimit-reset-requests": "24h0m0s"}),
            86400,
        )

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
            "現在", None, ("一", "二", "三", "四"), request_source="text"
        )

    async def test_busy_text_gate_falls_back_without_building_a_queue(self) -> None:
        service = GroqModerationService(
            "not-a-real-key",
            ModerationEngine.from_json(RULES_PATH),
            text_interval_seconds=60,
        )
        empty_payload = {
            "discrimination": {"detected": False, "confidence": 1, "reason": ""},
            "cynicism": {"detected": False, "confidence": 1, "reason": ""},
            "sexual_content": {"detected": False, "confidence": 1, "reason": ""},
            "sensitive_term": {"detected": False, "confidence": 1, "reason": ""},
            "drug_content": {"detected": False, "confidence": 1, "reason": ""},
        }
        service._post_chat = AsyncMock(return_value=empty_payload)

        await service.analyze("一件目の中立的な文")
        await service.analyze("二件目の中立的な文")

        self.assertEqual(service._post_chat.await_count, 1)
        self.assertEqual(service._quota_skipped_text_requests, 1)

    async def test_text_and_voice_have_independent_admission_slots(self) -> None:
        service = GroqModerationService(
            "not-a-real-key",
            ModerationEngine.from_json(RULES_PATH),
            text_interval_seconds=60,
            voice_interval_seconds=60,
        )
        empty_payload = {
            "discrimination": {"detected": False, "confidence": 1, "reason": ""},
            "cynicism": {"detected": False, "confidence": 1, "reason": ""},
            "sexual_content": {"detected": False, "confidence": 1, "reason": ""},
            "sensitive_term": {"detected": False, "confidence": 1, "reason": ""},
            "drug_content": {"detected": False, "confidence": 1, "reason": ""},
        }
        service._post_chat = AsyncMock(return_value=empty_payload)

        await service.analyze("通常テキスト")
        await service.analyze("VC文字起こし", request_source="voice")

        self.assertEqual(service._post_chat.await_count, 2)
        self.assertEqual(
            service._post_chat.await_args_list[1].kwargs["request_source"], "voice"
        )

    async def test_primary_text_429_uses_20b_fallback(self) -> None:
        service = self.make_service()
        category_payload = {
            "discrimination": {"detected": True, "confidence": 91, "reason": "排除です。"},
            "cynicism": {"detected": False, "confidence": 1, "reason": ""},
            "sexual_content": {"detected": False, "confidence": 1, "reason": ""},
            "sensitive_term": {"detected": False, "confidence": 1, "reason": ""},
            "drug_content": {"detected": False, "confidence": 1, "reason": ""},
        }
        service._request_json = AsyncMock(
            side_effect=[
                _RetryableApiError("tokens", status=429, retry_after_seconds=30),
                {"choices": [{"message": {"content": json.dumps(category_payload)}}]},
            ]
        )

        payload = await service._post_chat("文脈依存の表現", None)

        self.assertEqual(payload, category_payload)
        calls = service._request_json.await_args_list
        self.assertEqual(calls[0].kwargs["json_body"]["model"], "openai/gpt-oss-120b")
        self.assertEqual(calls[1].kwargs["json_body"]["model"], "openai/gpt-oss-20b")
        self.assertEqual(service._fallback_text_model_requests, 1)

    async def test_voice_uses_compact_safeguard_best_effort_schema(self) -> None:
        service = self.make_service()
        empty_payload = {
            "discrimination": {"detected": False, "confidence": 1, "reason": ""},
            "cynicism": {"detected": False, "confidence": 1, "reason": ""},
            "sexual_content": {"detected": False, "confidence": 1, "reason": ""},
            "sensitive_term": {"detected": False, "confidence": 1, "reason": ""},
            "drug_content": {"detected": False, "confidence": 1, "reason": ""},
        }
        service._request_json = AsyncMock(
            return_value={"choices": [{"message": {"content": json.dumps(empty_payload)}}]}
        )

        await service._post_chat("VC文字起こし", None, request_source="voice")

        body = service._request_json.await_args.kwargs["json_body"]
        self.assertEqual(body["model"], "openai/gpt-oss-safeguard-20b")
        self.assertEqual(body["messages"][0]["content"], service._VOICE_SYSTEM_PROMPT)
        self.assertFalse(body["response_format"]["json_schema"]["strict"])

    async def test_busy_audio_gate_keeps_only_one_bounded_api_waiter(self) -> None:
        fallback = AsyncMock()
        fallback.transcribe_wav.return_value = "ローカル文字起こし"
        service = GroqModerationService(
            "not-a-real-key",
            ModerationEngine.from_json(RULES_PATH),
            audio_interval_seconds=0.01,
            local_speech_transcriber=fallback,
        )
        service._post_audio = AsyncMock(return_value={"text": "外部文字起こし"})

        first = await service.transcribe_wav(b"RIFF-one")
        waiting = asyncio.create_task(service.transcribe_wav(b"RIFF-two"))
        await asyncio.sleep(0)
        third = await service.transcribe_wav(b"RIFF-three")
        second = await waiting

        self.assertEqual(first, "外部文字起こし")
        self.assertEqual(second, "外部文字起こし")
        self.assertEqual(third, "ローカル文字起こし")
        self.assertEqual(service._post_audio.await_count, 2)
        self.assertEqual(service._quota_skipped_audio_requests, 1)

    async def test_audio_waiter_does_not_call_api_during_new_cooldown(self) -> None:
        fallback = AsyncMock()
        fallback.transcribe_wav.return_value = "ローカル文字起こし"
        service = GroqModerationService(
            "not-a-real-key",
            ModerationEngine.from_json(RULES_PATH),
            audio_interval_seconds=0.01,
            local_speech_transcriber=fallback,
        )
        service._post_audio = AsyncMock(return_value={"text": "外部文字起こし"})

        await service.transcribe_wav(b"RIFF-one")
        waiting = asyncio.create_task(service.transcribe_wav(b"RIFF-two"))
        await asyncio.sleep(0)
        service._audio_cooldown_until = float("inf")

        self.assertEqual(await waiting, "ローカル文字起こし")
        self.assertEqual(service._post_audio.await_count, 1)

    def test_low_confidence_and_no_speech_segments_are_removed(self) -> None:
        payload = {
            "text": "fallback text",
            "segments": [
                {"text": "信頼できる部分", "avg_logprob": -0.1, "no_speech_prob": 0.1},
                {"text": "小さい声", "avg_logprob": -1.35, "no_speech_prob": 0.91},
                {"text": "低信頼", "avg_logprob": -1.5, "no_speech_prob": 0.1},
                {"text": "無音推定", "avg_logprob": -0.1, "no_speech_prob": 0.95},
            ],
        }
        service = self.make_service()
        self.assertEqual(
            service._trusted_transcript(payload),
            "信頼できる部分 小さい声",
        )

    async def test_voice_aliases_recover_common_short_transcription_errors(self) -> None:
        service = self.make_service()
        service._post_chat = AsyncMock()
        cases = {
            "魚": {"cynicism"},
            "市ね": {"cynicism"},
            "害児": {"discrimination"},
            "了解目う": {"cynicism"},
            "ドアー": {"cynicism"},
        }
        for transcript, expected in cases.items():
            with self.subTest(transcript=transcript):
                result = await service.analyze(transcript, request_source="voice")
                self.assertEqual(
                    {item.category for item in result.detections},
                    expected,
                )
        service._post_chat.assert_not_awaited()

    async def test_voice_only_alias_does_not_change_text_messages(self) -> None:
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

        result = await service.analyze("魚", request_source="text")

        self.assertFalse(result.detected)
        service._post_chat.assert_awaited_once()

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
        ):
            self.assertIn(phrase, prompt)

    def test_speech_prompt_avoids_bias_toward_moderation_vocabulary(self) -> None:
        prompt = GroqModerationService._SPEECH_PROMPT
        self.assertIn("実際に聞こえた内容だけ", prompt)
        self.assertIn("推測で語を補わず", prompt)
        for phrase in ("うお", "どわー", "めう", "ADHD"):
            self.assertNotIn(phrase, prompt)


if __name__ == "__main__":
    unittest.main()
