#!/usr/bin/env python3
from __future__ import annotations

import logging
from pathlib import Path
import sys

from discord_moderation_bot.bot import create_client
from discord_moderation_bot.config import BotConfig, ConfigError
from discord_moderation_bot.engine import ModerationEngine, RuleConfigurationError
from discord_moderation_bot.service import GroqModerationService, LocalModerationService
from discord_moderation_bot.speech import VoskSpeechTranscriber


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("discord_moderation_bot")

    try:
        config = BotConfig.from_env()
        engine = ModerationEngine.from_json(config.rules_path)
    except (ConfigError, RuleConfigurationError) as exc:
        logger.error("Configuration error: %s", exc)
        return 2

    if config.moderation_backend == "groq":
        local_speech = VoskSpeechTranscriber(
            Path(__file__).resolve().parent
            / "models"
            / "vosk-model-small-ja-0.22"
        )
        service = GroqModerationService(
            config.groq_api_key or "",
            engine,
            text_model=config.groq_text_model,
            fallback_text_model=config.groq_fallback_text_model,
            voice_text_model=config.groq_voice_text_model,
            speech_model=config.groq_speech_model,
            confidence_threshold=config.groq_confidence_threshold,
            cynicism_confidence_threshold=(
                config.groq_cynicism_confidence_threshold
            ),
            timeout_seconds=config.groq_timeout_seconds,
            voice_analysis_interval_seconds=(
                config.groq_voice_analysis_interval_seconds
            ),
            local_speech_transcriber=local_speech,
        )
    else:
        service = LocalModerationService(engine)

    client = create_client(config, service)
    try:
        client.run(config.token, log_handler=None)
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    return 0


if __name__ == "__main__":
    sys.exit(main())
