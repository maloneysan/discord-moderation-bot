# Discordモデレーション検知Bot

Botが参加しているすべてのDiscordサーバーで新規・編集テキストを監視し、冷笑、差別、性的表現、サーバー指定要注意語、薬物関連の可能性があるときに注意喚起するBotです。GroqのGPT-OSS 120Bで日本語を含む多言語、文脈、婉曲表現、皮肉、隠語、下ネタ、薬物の隠語や勧誘を判定し、既知表現は端末内JSONルールでも補完します。返信元に加えて同じチャンネルの直前3件を最大3分だけメモリ内で参照し、主語省略や複数投稿にまたがる言い回しも判断します。

任意でVCも監視できます。VC音声はメモリ内で短いWAVへ変換し、Groq Whisper Large V3で多言語文字起こしして同じ文脈判定へ渡します。通知には発言者の表示名、該当発言（最大240文字）、問題だった意味・作用を記載します。本文・音声・文字起こしはBotのファイルやDBへ保存しません。詳しい動作と制約は[仕様書](docs/SPECIFICATION.md)を参照してください。

## 必要環境

- Python 3.9以上
- Discord BotアプリケーションとBotトークン
- 無料GroqアカウントとAPIキー
- Botを追加できる各Discordサーバーでの「サーバーを管理」権限
- VC監視時のみ、音声受信依存パッケージとネイティブOpus

## Discord側の設定

1. [Discord Developer Portal](https://discord.com/developers/applications)で対象Applicationを開きます。
2. `Bot`ページの`Privileged Gateway Intents`で`Message Content Intent`を有効にします。
3. `Installation`または`OAuth2 > URL Generator`で`bot`と`applications.commands`スコープを選びます。
4. 次の権限だけを選択します。

   - View Channels
   - Send Messages
   - Read Message History
   - Send Messages in Threads
   - Connect（VC監視を使う場合のみ）

管理者権限、Manage Messages、メンバー管理権限、Speakは不要です。チャンネルごとの権限上書きでも、Botが閲覧・送信できることを確認してください。

Botが要求・使用する権限は`View Channels`、`Send Messages`、`Read Message History`、`Send Messages in Threads`、`Connect`だけです。`Message Content Intent`はOAuth権限ではなくDeveloper Portalの特権Intentです。`applications.commands`はスラッシュコマンド用スコープです。Discord上では`/permissions`で現在のチャンネルの実効権限を確認できます。

現在のBotを別サーバーへ追加する場合は、追加先を管理できる人が次のリンクを開き、サーバーを選びます。

<https://discord.com/oauth2/authorize?client_id=1541913684043890819&permissions=274879024128&integration_type=0&scope=bot%20applications.commands>

リンクを他人へ教えなくても、Applicationが公開Bot設定なら追加先サーバーを選べます。現在の自動起動設定では、Botを追加したサーバーは自動的にテキスト・VC監視対象になります。

## インストール

テキスト監視のみの場合:

```bash
cd DiscordModerationBot
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

VC監視も使う場合:

```bash
brew install opus
source .venv/bin/activate
python3 -m pip install -r requirements-voice.txt
```

`opus`はDiscord音声をPCMへ復号するために必要な無料のオープンソースライブラリです。

## 環境変数

トークンをファイルやコマンド履歴へ残さず、利用中のシェルまたはOSのシークレット管理機能から設定してください。

```bash
export DISCORD_BOT_TOKEN='Botトークン'
export DISCORD_GUILD_IDS='サーバーID1,サーバーID2'
export MONITOR_ALL_GUILDS='true'
export MONITORED_CHANNEL_IDS='チャンネルID1,チャンネルID2'
export ALERT_CHANNEL_IDS='サーバーID1:通知チャンネルID1,サーバーID2:通知チャンネルID2'
export MODERATION_BACKEND='groq'
export GROQ_API_KEY='GroqのAPIキー'
export GROQ_TEXT_MODEL='openai/gpt-oss-120b'
export GROQ_SPEECH_MODEL='whisper-large-v3'
export GROQ_CONFIDENCE_THRESHOLD='50'
export GROQ_CYNICISM_CONFIDENCE_THRESHOLD='80'
export CONTEXT_MESSAGE_COUNT='3'
export CONTEXT_TTL_SECONDS='180'
```

- `DISCORD_BOT_TOKEN`は必須です。`MONITOR_ALL_GUILDS=true`なら`DISCORD_GUILD_IDS`は省略でき、Bot参加中の全サーバーを対象にします。
- `MONITORED_CHANNEL_IDS`を省略すると、指定サーバー内でBotが閲覧できる全テキストチャンネルとスレッドを監視します。
- `ALERT_CHANNEL_IDS`は`サーバーID:チャンネルID`形式です。指定がないサーバーでは元投稿へ返信します。
- 旧版の`DISCORD_GUILD_ID`と`ALERT_CHANNEL_ID`も、1サーバー運用に限り互換対応します。
- APIキーは[Groq Console](https://console.groq.com/keys)で作成します。無料枠上限や一時障害時はテキストだけローカルルールへ自動退避します。
- 会話文脈には同じチャンネルの直前メッセージだけを使い、投稿者名は付けません。既定は3件・180秒で、ディスクやDBへ保存せず、期限切れとBot終了時に破棄します。過去の発言だけを理由に現在の無害な投稿を検知しないよう、AIへ現在メッセージだけを分類させます。

VC監視を有効にする場合は追加します。

```bash
export VOICE_ENABLED='true'
export VOICE_AUTO_JOIN='true'
export VOICE_CHANNEL_IDS='サーバーID1:VCチャンネルID1,サーバーID2:VCチャンネルID2'
export VOICE_ALERT_CHANNEL_IDS='サーバーID1:通知チャンネルID1,サーバーID2:通知チャンネルID2'
export VOICE_CHUNK_SECONDS='10'
export VOICE_MIN_RMS='80'
export VOICE_MIN_UTTERANCE_MS='180'
```

`VOICE_AUTO_JOIN=true`では、各サーバーで人が入ったVCへBotが自動参加します。監視中VCに人がいる間はそのVCを維持し、空になると利用中で最も人数の多いVCへ移動します。Discordの仕様上、Botが同時参加できるVCは1サーバーにつき1つなので、同一サーバー内の複数VCを同時監視はできません。Botが参加していないVCの音声は取得できず、検知もできません。

BotはVC参加時、VC内テキストまたは通知先へ音声認識中であることと非保存方針を案内します。発話は最大10秒、Discordの発話終了イベント、または音声後の無音検出時に送信します。明瞭な短文を拾うためRMS閾値を80、最短発話を180msへ緩和しました。Whisper Large V3へ日本語を明示し、`verbose_json`のうち`avg_logprob < -1.0`または`no_speech_prob > 0.8`だけを低品質として除外します。[Groq公式の音声品質メタデータ解説](https://console.groq.com/docs/speech-to-text)

`/vc_status`では認識済みチャンク、無音・低品質除外、最終認識時刻、直近エラーを確認できます。運用ログには文字起こし本文を残さず、チャンクが認識・判定まで到達したかだけを記録します。

VC通知には発言者のサーバー表示名、音声認識した該当発言（最大240文字）、「能力への見下し」「失敗の嘲笑」「集団の排除」など問題だった内容を具体化した短い要約を載せます。該当発言は改行とDiscordのメンション・Markdown記号を無効化して表示し、音声や文字起こしをDB・ファイル・ログへ保存しません。通知先未指定時は、Botが送信できるシステムチャンネルまたは最初のテキストチャンネルを自動選択します。20秒以内の同一話者・同一分類は1回へまとめます。音声受信・接続・認識が止まった場合は、同じ障害を5分間まとめて管理用通知先へ知らせ、テキスト監視は継続します。

`.env.example`は変数名の確認用で、Bot自身は`.env`を読み込みません。

## 起動

```bash
python3 run.py
```

終了は`Ctrl+C`です。一時切断時は`discord.py`が再接続を試みます。VC初期化や権限に失敗しても、テキスト監視は継続します。

### このMacで安全に起動する

[start_bot_secure.command](start_bot_secure.command)をFinderからダブルクリックすると、入力を表示せずBotトークンとGroq APIキーを尋ね、その実行中だけ環境変数へ渡します。続けて監視サーバー、専用通知先、VC利用の有無を対話入力できます。秘密値はファイルや履歴へ保存しません。

サーバーIDは任意です。`MONITOR_ALL_GUILDS=true`でBot参加中の全サーバーを監視するため、特定サーバーだけを設定資料として残したい場合にカンマ区切りで入力します。ターミナルを閉じる、Macを終了する、または`Ctrl+C`を押すとBotも停止します。

### macOSログイン時に自動起動する

最初に`config/autostart.env.example`を`config/autostart.env`へコピーし、サーバー・チャンネル設定を編集します。その後Finderから[install_autostart.command](install_autostart.command)をダブルクリックします。BotトークンとGroq APIキーを1回入力すると、平文ファイルではなくmacOSキーチェーンへ保存し、ユーザーのログイン時にLaunchAgentがBotを起動します。Botが異常終了した場合も30秒以上空けて再起動します。macOSのバックグラウンド実行制限を避けるため、実行用コピーは`~/Library/Application Support/DiscordModerationBot`へ配置されます。

```bash
cp config/autostart.env.example config/autostart.env
```

自動起動対象のサーバーやVCは[config/autostart.env](config/autostart.env)で設定します。このファイルにトークンは書かず、Git管理もしません。現在は参加中の全サーバー、閲覧可能な全テキストチャンネル、VC自動参加が有効です。コードや自動起動設定を変更した後は、インストーラーをもう一度実行すると実行用コピーが更新されます。既存のキーチェーントークンがある場合、再入力は求めません。

状態と内容を確認する場合:

```bash
launchctl print "gui/$(id -u)/io.github.maloneysan.discord-moderation-bot"
tail -n 50 "$HOME/Library/Application Support/DiscordModerationBot/logs/bot.log"
tail -n 50 "$HOME/Library/Application Support/DiscordModerationBot/logs/bot-error.log"
```

自動起動とキーチェーン内のトークンを削除する場合は[uninstall_autostart.command](uninstall_autostart.command)を実行します。PCの電源が入っていても、ユーザーがmacOSへログインしキーチェーンが利用可能になるまではBotは起動しません。

## ライセンスとセキュリティ

ソースコードは[MIT License](LICENSE)で公開します。脆弱性や秘密情報漏えいにつながる問題は、実際のトークンや投稿内容を公開Issueへ貼らず、[SECURITY.md](SECURITY.md)の手順で報告してください。

## スラッシュコマンド

Discordの入力欄で`/`を入力すると表示されます。

| コマンド | 動作 | 権限 |
| --- | --- | --- |
| `/vc_join` | 実行者がいるVCへBotを参加させる | サーバーを管理 |
| `/vc_leave` | VCから退出し、そのサーバーの自動参加を一時停止 | サーバーを管理 |
| `/vc_auto enabled:true/false` | VC自動参加を再開・停止（停止時は現在のVCから退出） | サーバーを管理 |
| `/vc_status` | VC、受信機能、認識件数、低品質除外、最終認識、直近エラーを表示 | 全員 |
| `/bot_status` | 稼働時間、判定・通知件数、会話文脈、API・再接続・VC状態を表示 | 全員 |
| `/permissions` | 現在のチャンネルでのBot実効権限を表示 | 全員 |
| `/alert_channel channel:#通知先` | テキスト・VCの検知通知先を設定して再起動後も維持 | サーバーを管理 |
| `/alert_channel_reset` | コマンド指定を解除し、環境設定または自動選択へ戻す | サーバーを管理 |
| `/alert_channel_status` | 現在適用されている通知先を表示 | サーバーを管理 |
| `/ping` | Botの応答速度を表示 | 全員 |
| `/moderation_help` | コマンド一覧を表示 | 全員 |

コマンドの応答は実行者だけに見える形式です。`/alert_channel`は同じサーバー内でBotが閲覧・送信できる通常テキストチャンネルだけを受け付け、テキストとVCの通知先を同時に変更します。設定ファイルにはサーバーIDとチャンネルIDだけを保存し、投稿内容や文字起こしは保存しません。`/vc_auto enabled:false`は現在のVCから退出して自動参加を一時停止し、`enabled:true`は利用者がいるVCへ即時参加します。利用者がいなければ待機し、次のVC参加イベントで追従します。一時停止はBot再起動で解除され、自動参加へ戻ります。

Developer PortalのBot説明にも、次の短縮一覧を掲載します。

`/vc_join /vc_leave /vc_auto /vc_status /bot_status /permissions /alert_channel /alert_channel_reset /alert_channel_status /ping /moderation_help`

## 判定ルール

ルールは[config/rules.json](config/rules.json)にあります。

- `threshold`: 通知する最低点（初期値80）
- `categories`: 内部分類と通知ラベル
- `exceptions`: 批判的な引用や反差別表現などの除外正規表現
- `rules`: ルールID、分類、点数、正規表現

GPT-OSS 120Bは現在メッセージ、返信元、直前3件の一時文脈から、保護属性、脆弱集団、隠語、婉曲表現、集団責任、劣等視、排除、権利否定、非人間化、侮辱、見下し、嘲笑などを多言語で判定します。冷笑は誤検知を抑えるため、原則として「識別可能な対象」と「明確な見下し・嘲笑」の両方があり、信頼度80以上の場合だけ通知します。普通の反対意見、訂正、驚き、愚痴、短い返事、友好的な軽口、対象のない「笑・草・w」は冷笑にしません。「女々しい」「男らしくない」「男のくせに泣くな」などの性別に基づく侮蔑は差別と冷笑の両方として扱います。発言スタイルから「女性らしい」などを推測するのではなく、明示的な性別侮蔑表現だけがローカル強ルールの対象です。

「うお」「うおw」「どわー」「クイヤ」と、語尾が「めう」で終わる投稿はサーバー固有の冷笑表現としてローカル安全網でも直接検知します。「めうという語尾について話す」のように末尾でない中立的説明、「うお座」「ドワーフ」は除外します。「クイヤ」は要望どおり、投稿内に表記が出た時点で検知します。

下ネタ、露骨な性的行為・身体部位・性的な冗談や婉曲表現は「性的表現」として検知します。医療・教育・安全・同意・報道の中立的文脈はAIで除外し、ローカル安全網でも「エッチング加工」「処女航海」「ちんちん電車」などの同音異義語を除外します。

`ADHD`は要望どおり、独立した英字語として投稿内に出た時点で「要注意語（ADHD）」として検知します。中立的な説明や自己申告も通知対象ですが、差別とは表示しません。`ADHDers`のように別の英単語の一部になっている場合はローカルルールでは検知しません。

違法・娯楽目的の薬物名、薬物乱用やオーバードーズ、製造・栽培、所持、売買、譲渡、勧誘、摂取方法、隠語を「薬物関連」として検知します。一般的な処方・治療、「ドラッグストア」、「OD缶」だけでは成立させません。サーバー方針として明示的な違法薬物名は報道・教育・防止・治療の文脈でも確認対象になります。

成立した分類ごとに、GPT-OSS 120Bが問題だった意味や会話上の作用を日本語1文で要約します。要約自体には原文、ユーザー名、ID、リンク、メンションを含めないよう指示し、Bot側でも改行・メンション文字・長さを安全化します。通知にはこれとは別に、実際の該当発言を最大240文字で明示します。API障害時は分類別の定型説明を使用します。

## 自動テスト

Discordへ接続せず実行できます。

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q discord_moderation_bot run.py tests scripts
```

### 精度評価

[config/evaluation_cases.json](config/evaluation_cases.json)には、実投稿でなく56件の人工例文を収録しています。明白な違反、婉曲・伏字、複数投稿文脈、複合分類、性別侮蔑、下ネタ、ADHD、薬物名・乱用・売買・隠語、サーバー固有語、引用・反差別・中立言及・一般的医療に加え、対象のない驚き・愚痴・友好的な軽口・通常の反対意見を同じ基準で測れます。結果には本文を出さず、集計値と不一致ケースIDだけを表示します。

```bash
# APIを消費しないローカル安全網の評価
python3 scripts/evaluate_moderation.py --backend local

# Groqを含む本番判定の評価（GROQ_API_KEYが必要）
python3 scripts/evaluate_moderation.py --backend groq
```

現在のローカル安全網は52件で適合率1.000、再現率0.900、F1 0.947です。文脈専用の差別・冷笑・性的婉曲表現・薬物隠語はAI主系で補完します。薬物の文脈付き隠語と「OD缶」の境界はGroq実APIの個別確認で2/2件一致しました。

## 実サーバー確認

安全なテスト専用チャンネル・VCで確認してください。実際の差別語を不必要に公開せず、VC参加者へ音声認識Botが参加することを事前に知らせてください。

1. 各設定済みサーバーの新規投稿と編集へ反応する。
2. 「草」だけ、属性への中立的言及、「ドワーフ」では通知しない。
3. 「うお」「うおw」「どわー」「クイヤ」、語尾「めう」、性別に基づく侮蔑と代表的な差別表現は分類ごとに1回だけ通知する。
4. 下ネタ・性的表現を「性的表現」、ADHDへの言及を「要注意語（ADHD）」として通知する。ADHDは差別と断定しない。
5. 違法薬物名、薬物乱用、売買・勧誘・摂取方法を「薬物関連」として通知し、通常の処方薬やドラッグストアだけでは通知しない。
6. スレッドとサーバー別専用通知チャンネルで通知できる。
7. 通知に発言者の表示名、該当発言（最大240文字）、問題点が載り、メンションは発生しない。
8. Bot、Webhook、DM、対象外チャンネルを無視する。
9. 人がVCへ入るとBotが自動参加し、空になると退出または別の利用中VCへ移動する。
10. `/vc_join`などのスラッシュコマンドが表示され、権限どおり操作できる。
11. 送信・接続権限を一時的に外してもプロセス全体は停止しない。

## プライバシーと制約

- 本文、返信元、直前3件の一時文脈は判定のためGroqへ送信します。VC音声チャンクも文字起こしのためGroqへ送信します。
- 本文、VC音声、文字起こしをBotのDB・ファイル・ログへ保存しません。Groqは通常推論を既定では保持しないと説明していますが、障害調査・不正利用調査では一時保持される可能性があります。
- 通知には発言者の現在の表示名、該当発言またはVC文字起こし（最大240文字）、「どの対象を、どのように扱ったことが問題か」を説明する要約が残ります。問題表現の再掲を管理者だけに限定したい場合は、`/alert_channel_set`で管理者専用チャンネルを指定してください。
- 無料枠の目安はテキスト1,000リクエスト/日、音声8時間/日です。上限、モデル変更、サービス停止により外部判定できない場合があります。
- 投稿削除、ミュート、BAN、自動処罰は行いません。
- VC受信はDiscordのDAVE暗号化対応が必要なため、受信前復号と復号待ちパケット破棄を実装した未マージの音声受信拡張を特定コミットに固定しています。Discord側の変更で動かなくなる可能性があります。
- AI判定と音声認識には聞き間違い、誤検知、見逃しがあります。「この世のすべて」を100%検知する保証はできません。通知を処罰の唯一の根拠にしないでください。
