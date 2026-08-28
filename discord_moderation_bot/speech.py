from __future__ import annotations

import asyncio
import importlib
import io
import json
import logging
from pathlib import Path
import threading
import wave


LOGGER = logging.getLogger(__name__)


class VoskSpeechTranscriber:
    """Optional, offline Japanese speech fallback that never stores audio or text."""

    def __init__(self, model_path: Path) -> None:
        self._model_path = model_path
        self._model = None
        self._model_lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self._model_path.is_dir()

    async def transcribe_wav(self, wav_audio: bytes) -> str:
        if not wav_audio or not self.available:
            return ""
        try:
            return await asyncio.to_thread(self._transcribe_sync, wav_audio)
        except Exception as exc:
            LOGGER.warning(
                "Local speech fallback failed (error=%s)", type(exc).__name__
            )
            return ""

    def _transcribe_sync(self, wav_audio: bytes) -> str:
        vosk = importlib.import_module("vosk")
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    vosk.SetLogLevel(-1)
                    self._model = vosk.Model(str(self._model_path))

        with wave.open(io.BytesIO(wav_audio), "rb") as wav_file:
            if (
                wav_file.getnchannels() != 1
                or wav_file.getsampwidth() != 2
                or wav_file.getframerate() != 16_000
            ):
                raise ValueError("local speech fallback requires 16 kHz mono PCM WAV")
            recognizer = vosk.KaldiRecognizer(self._model, 16_000)
            while True:
                frames = wav_file.readframes(4_000)
                if not frames:
                    break
                recognizer.AcceptWaveform(frames)

        payload = json.loads(recognizer.FinalResult())
        text = payload.get("text", "") if isinstance(payload, dict) else ""
        return text.strip() if isinstance(text, str) else ""
