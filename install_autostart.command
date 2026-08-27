#!/bin/zsh

set -eu

BOT_DIR="${0:A:h}"
LABEL="io.github.maloneysan.discord-moderation-bot"
LEGACY_LABEL="com.abeyuri.discord-moderation-bot"
SERVICE="DiscordModerationBotToken"
GROQ_SERVICE="DiscordModerationBotGroqApiKey"
ACCOUNT="$(/usr/bin/id -un)"
DOMAIN="gui/$(/usr/bin/id -u)"
SOURCE_PLIST="$BOT_DIR/launchd/$LABEL.plist"
DESTINATION_DIR="$HOME/Library/LaunchAgents"
DESTINATION_PLIST="$DESTINATION_DIR/$LABEL.plist"
RUNTIME_DIR="$HOME/Library/Application Support/DiscordModerationBot"

print "Discord Botの自動起動を設定します。"
if [[ ! -r "$BOT_DIR/config/autostart.env" ]]; then
  print "設定エラー: config/autostart.env がありません。"
  print "config/autostart.env.example をコピーして設定してください。"
  exit 2
fi
if /usr/bin/security find-generic-password -a "$ACCOUNT" -s "$SERVICE" >/dev/null 2>&1; then
  print "既存のキーチェーントークンを使用します。"
else
  print "次の入力欄にBotトークンを入力してください（入力内容は表示されません）。"
  print "トークンは平文ファイルではなくmacOSキーチェーンへ保存されます。"
  /usr/bin/security add-generic-password \
    -a "$ACCOUNT" \
    -s "$SERVICE" \
    -l "Discord Moderation Bot Token" \
    -U \
    -T /usr/bin/security \
    -w
fi

if /usr/bin/grep -q '^MODERATION_BACKEND=groq$' "$BOT_DIR/config/autostart.env"; then
  if /usr/bin/security find-generic-password -a "$ACCOUNT" -s "$GROQ_SERVICE" >/dev/null 2>&1; then
    print "既存のGroq APIキーを使用します。"
  else
    print "次の入力欄にGroq APIキーを入力してください（入力内容は表示されません）。"
    print "APIキーは平文ファイルではなくmacOSキーチェーンへ保存されます。"
    /usr/bin/security add-generic-password \
      -a "$ACCOUNT" \
      -s "$GROQ_SERVICE" \
      -l "Discord Moderation Bot Groq API Key" \
      -U \
      -T /usr/bin/security \
      -w
  fi
fi

/bin/mkdir -p "$DESTINATION_DIR" "$RUNTIME_DIR/logs"
/usr/bin/ditto "$BOT_DIR/.venv" "$RUNTIME_DIR/.venv"
/usr/bin/ditto "$BOT_DIR/discord_moderation_bot" "$RUNTIME_DIR/discord_moderation_bot"
/usr/bin/ditto "$BOT_DIR/config" "$RUNTIME_DIR/config"
/usr/bin/ditto "$BOT_DIR/scripts" "$RUNTIME_DIR/scripts"
/bin/cp "$BOT_DIR/run.py" "$RUNTIME_DIR/run.py"
/bin/chmod 700 "$RUNTIME_DIR/scripts/run_from_keychain.zsh"
/bin/cp "$SOURCE_PLIST" "$DESTINATION_PLIST"
/usr/libexec/PlistBuddy -c "Set :ProgramArguments:0 $RUNTIME_DIR/scripts/run_from_keychain.zsh" "$DESTINATION_PLIST"
/usr/libexec/PlistBuddy -c "Set :WorkingDirectory $RUNTIME_DIR" "$DESTINATION_PLIST"
/usr/libexec/PlistBuddy -c "Set :StandardOutPath $RUNTIME_DIR/logs/bot.log" "$DESTINATION_PLIST"
/usr/libexec/PlistBuddy -c "Set :StandardErrorPath $RUNTIME_DIR/logs/bot-error.log" "$DESTINATION_PLIST"
/bin/chmod 600 "$DESTINATION_PLIST"

/bin/launchctl bootout "$DOMAIN/$LEGACY_LABEL" >/dev/null 2>&1 || true
/bin/rm -f "$DESTINATION_DIR/$LEGACY_LABEL.plist"
/bin/launchctl bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
if ! /bin/launchctl bootstrap "$DOMAIN" "$DESTINATION_PLIST"; then
  print "LaunchAgentの再登録を待って再試行します。"
  /bin/sleep 2
  /bin/launchctl bootstrap "$DOMAIN" "$DESTINATION_PLIST"
fi
/bin/launchctl enable "$DOMAIN/$LABEL"
/bin/launchctl kickstart -k "$DOMAIN/$LABEL"

print
print "設定完了: Botを起動し、次回ログイン時の自動起動も有効にしました。"
print "状態確認: launchctl print $DOMAIN/$LABEL"
print "ログ: $RUNTIME_DIR/logs/bot.log"
read -r "?Enterキーで閉じます。"
