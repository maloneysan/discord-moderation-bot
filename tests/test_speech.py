from pathlib import Path
import unittest

from discord_moderation_bot.speech import VoskSpeechTranscriber


class VoskSpeechTranscriberTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_model_returns_empty_without_retaining_audio(self) -> None:
        transcriber = VoskSpeechTranscriber(Path("/definitely/missing/model"))
        audio = b"RIFF-private-audio"

        transcript = await transcriber.transcribe_wav(audio)

        self.assertEqual(transcript, "")
        self.assertNotIn("private-audio", repr(transcriber.__dict__))


if __name__ == "__main__":
    unittest.main()
