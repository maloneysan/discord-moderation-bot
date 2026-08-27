#!/bin/zsh

set -u

BOT_DIR="${0:A:h}"
cd "$BOT_DIR" || exit 1

if [[ ! -x ".venv/bin/python" ]]; then
  print "起動エラー: .venv が見つかりません。READMEのセットアップ手順を確認してください。"
  read -r "?Enterキーで閉じます。"
  exit 1
fi

read -s "DISCORD_BOT_TOKEN?Bot token（入力は表示されません）: "
print

if [[ -z "$DISCORD_BOT_TOKEN" ]]; then
  print "起動を中止しました: Bot tokenが空です。"
  read -r "?Enterキーで閉じます。"
  exit 1
fi

export DISCORD_BOT_TOKEN

read -s "GROQ_API_KEY?Groq API key（入力は表示されません）: "
print
if [[ -z "$GROQ_API_KEY" ]]; then
  print "起動を中止しました: Groq API keyが空です。"
  read -r "?Enterキーで閉じます。"
  exit 1
fi
export GROQ_API_KEY
export MODERATION_BACKEND="groq"
export GROQ_TEXT_MODEL="openai/gpt-oss-120b"
export GROQ_SPEECH_MODEL="whisper-large-v3"
export GROQ_CONFIDENCE_THRESHOLD="50"

read -r "INPUT_GUILD_IDS?監視するサーバーID（任意、複数はカンマ区切り）: "
if [[ -n "$INPUT_GUILD_IDS" ]]; then
  export DISCORD_GUILD_IDS="$INPUT_GUILD_IDS"
fi
export MONITOR_ALL_GUILDS="true"

read -r "INPUT_ALERT_CHANNEL_IDS?専用通知先（サーバーID:チャンネルID、未指定は元投稿へ返信）: "
if [[ -n "$INPUT_ALERT_CHANNEL_IDS" ]]; then
  export ALERT_CHANNEL_IDS="$INPUT_ALERT_CHANNEL_IDS"
fi

read -r "ENABLE_VOICE?VC監視を有効にしますか？ [y/N]: "
if [[ "${ENABLE_VOICE:l}" == "y" || "${ENABLE_VOICE:l}" == "yes" ]]; then
  read -r "INPUT_VOICE_CHANNEL_IDS?監視VC（サーバーID:VCチャンネルID）: "
  read -r "INPUT_VOICE_ALERT_CHANNEL_IDS?VC通知先（サーバーID:テキストチャンネルID）: "
  export VOICE_ENABLED="true"
  export VOICE_AUTO_JOIN="true"
  export VOICE_CHANNEL_IDS="$INPUT_VOICE_CHANNEL_IDS"
  export VOICE_ALERT_CHANNEL_IDS="$INPUT_VOICE_ALERT_CHANNEL_IDS"
  export VOICE_CHUNK_SECONDS="10"
else
  export VOICE_ENABLED="false"
fi

exec .venv/bin/python run.py
