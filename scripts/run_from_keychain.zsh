#!/bin/zsh

set -eu

BOT_DIR="${0:A:h:h}"
CONFIG_FILE="$BOT_DIR/config/autostart.env"
KEYCHAIN_SERVICE="DiscordModerationBotToken"
GROQ_KEYCHAIN_SERVICE="DiscordModerationBotGroqApiKey"
KEYCHAIN_ACCOUNT="$(/usr/bin/id -un)"

if [[ ! -x "$BOT_DIR/.venv/bin/python" ]]; then
  print -u2 "Bot起動エラー: Python仮想環境がありません"
  exit 2
fi
if [[ ! -r "$CONFIG_FILE" ]]; then
  print -u2 "Bot起動エラー: 自動起動設定がありません"
  exit 2
fi

while IFS='=' read -r key value; do
  key="${key//$'\r'/}"
  value="${value//$'\r'/}"
  [[ -z "$key" || "$key" == \#* ]] && continue
  case "$key" in
    DISCORD_GUILD_IDS|MONITOR_ALL_GUILDS|MONITORED_CHANNEL_IDS|ALERT_CHANNEL_IDS|VOICE_ENABLED|VOICE_AUTO_JOIN|VOICE_CHANNEL_IDS|VOICE_ALERT_CHANNEL_IDS|MODERATION_BACKEND|GROQ_TEXT_MODEL|GROQ_FALLBACK_TEXT_MODEL|GROQ_VOICE_TEXT_MODEL|GROQ_SPEECH_MODEL|GROQ_CONFIDENCE_THRESHOLD|GROQ_CYNICISM_CONFIDENCE_THRESHOLD|GROQ_TIMEOUT_SECONDS|GROQ_TEXT_ANALYSIS_INTERVAL_SECONDS|GROQ_VOICE_ANALYSIS_INTERVAL_SECONDS|GROQ_AUDIO_TRANSCRIPTION_INTERVAL_SECONDS|VOICE_CHUNK_SECONDS|CONTEXT_MESSAGE_COUNT|CONTEXT_TTL_SECONDS|VOICE_MIN_RMS|VOICE_MIN_UTTERANCE_MS)
      export "$key=$value"
      ;;
  esac
done < "$CONFIG_FILE"

DISCORD_BOT_TOKEN="$(
  /usr/bin/security find-generic-password \
    -a "$KEYCHAIN_ACCOUNT" \
    -s "$KEYCHAIN_SERVICE" \
    -w
)"
if [[ -z "$DISCORD_BOT_TOKEN" ]]; then
  print -u2 "Bot起動エラー: キーチェーンのトークンが空です"
  exit 2
fi

export DISCORD_BOT_TOKEN
if [[ "${MODERATION_BACKEND:-local}" == "groq" ]]; then
  GROQ_API_KEY="$(
    /usr/bin/security find-generic-password \
      -a "$KEYCHAIN_ACCOUNT" \
      -s "$GROQ_KEYCHAIN_SERVICE" \
      -w
  )"
  if [[ -z "$GROQ_API_KEY" ]]; then
    print -u2 "Bot起動エラー: キーチェーンのGroq APIキーが空です"
    exit 2
  fi
  export GROQ_API_KEY
fi
export PYTHONUNBUFFERED=1
cd "$BOT_DIR"
exec "$BOT_DIR/.venv/bin/python" "$BOT_DIR/run.py"
