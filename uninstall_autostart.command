#!/bin/zsh

set -u

LABEL="io.github.maloneysan.discord-moderation-bot"
LEGACY_LABEL="com.abeyuri.discord-moderation-bot"
SERVICE="DiscordModerationBotToken"
GROQ_SERVICE="DiscordModerationBotGroqApiKey"
ACCOUNT="$(/usr/bin/id -un)"
DOMAIN="gui/$(/usr/bin/id -u)"
DESTINATION_PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
RUNTIME_DIR="$HOME/Library/Application Support/DiscordModerationBot"

/bin/launchctl bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
/bin/launchctl bootout "$DOMAIN/$LEGACY_LABEL" >/dev/null 2>&1 || true
if [[ -f "$DESTINATION_PLIST" ]]; then
  /bin/rm "$DESTINATION_PLIST"
fi
LEGACY_PLIST="$HOME/Library/LaunchAgents/$LEGACY_LABEL.plist"
if [[ -f "$LEGACY_PLIST" ]]; then
  /bin/rm "$LEGACY_PLIST"
fi
/usr/bin/security delete-generic-password \
  -a "$ACCOUNT" \
  -s "$SERVICE" >/dev/null 2>&1 || true
/usr/bin/security delete-generic-password \
  -a "$ACCOUNT" \
  -s "$GROQ_SERVICE" >/dev/null 2>&1 || true
if [[ -d "$RUNTIME_DIR" ]]; then
  /bin/rm -rf "$RUNTIME_DIR"
fi

print "自動起動設定とキーチェーン内のBotトークン・Groq APIキーを削除しました。"
read -r "?Enterキーで閉じます。"
