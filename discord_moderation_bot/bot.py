from __future__ import annotations

import logging
import time
from typing import Optional, Set

import discord
from discord import app_commands

from .config import BotConfig
from .context import ConversationContextBuffer
from .destinations import AlertChannelStore
from .policy import (
    MessageContext,
    build_alert_text,
    build_voice_alert_text,
    should_monitor_message,
)
from .state import AlertRegistry
from .service import ModerationService
from .models import CategoryDetection
from .voice import VoiceModerationManager


LOGGER = logging.getLogger(__name__)


class ModerationClient(discord.Client):
    def __init__(self, config: BotConfig, service: ModerationService) -> None:
        intents = discord.Intents.none()
        intents.guilds = True
        intents.guild_messages = True
        intents.message_content = True
        intents.voice_states = config.voice_enabled
        super().__init__(intents=intents, max_messages=100)
        self._config = config
        self._service = service
        self._alerts = AlertRegistry()
        self._context = ConversationContextBuffer(
            max_messages=config.context_message_count,
            ttl_seconds=config.context_ttl_seconds,
        )
        self._started_at = time.monotonic()
        self._messages_analyzed = 0
        self._contextual_analyses = 0
        self._text_alerts_sent = 0
        self._last_resumed_at: Optional[float] = None
        self._alert_channels = AlertChannelStore(config.runtime_state_path)
        self._voice = VoiceModerationManager(
            self,
            config,
            service,
            self._send_voice_alert,
            self._send_voice_notice,
        )
        self.tree = app_commands.CommandTree(self)
        self._synced_command_guilds: Set[int] = set()
        self._register_application_commands()

    async def on_ready(self) -> None:
        available = []
        unavailable = []
        if self._config.monitor_all_guilds:
            available = [guild.id for guild in self.guilds]
        else:
            for guild_id in sorted(self._config.guild_ids):
                if self.get_guild(guild_id) is None:
                    unavailable.append(guild_id)
                else:
                    available.append(guild_id)
        if unavailable:
            LOGGER.warning(
                "Bot is connected, but target guilds are unavailable (count=%s)",
                len(unavailable),
            )
        LOGGER.info(
            "Moderation Bot is ready (user_id=%s, guild_count=%s)",
            self.user.id if self.user else "unknown",
            len(available),
        )
        await self._voice.start()
        for guild_id in available:
            guild = self.get_guild(guild_id)
            if guild is not None:
                await self._sync_commands_for_guild(guild)
                self._audit_guild_permissions(guild)

    async def on_guild_join(self, guild: discord.Guild) -> None:
        if not self._config.is_guild_monitored(guild.id):
            return
        await self._sync_commands_for_guild(guild)
        await self._voice.sync_guild(guild)

    async def on_guild_remove(self, guild: discord.Guild) -> None:
        self._synced_command_guilds.discard(guild.id)
        try:
            self._alert_channels.clear(guild.id)
        except OSError as exc:
            LOGGER.warning(
                "Could not clear departed guild alert destination (guild_id=%s, error=%s)",
                guild.id,
                type(exc).__name__,
            )

    async def on_disconnect(self) -> None:
        LOGGER.warning("Disconnected from Discord; the client will attempt to reconnect")

    async def on_resumed(self) -> None:
        self._last_resumed_at = time.monotonic()
        LOGGER.info("Discord session resumed")

    async def on_message(self, message: discord.Message) -> None:
        await self._process_message(message)

    async def on_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        await self._voice.handle_voice_state_update(member, before, after)

    async def on_raw_message_edit(self, payload: discord.RawMessageUpdateEvent) -> None:
        if payload.guild_id is None or not self._config.is_guild_monitored(
            payload.guild_id
        ):
            return

        try:
            channel = await self._resolve_channel(payload.channel_id)
            if channel is None or not hasattr(channel, "fetch_message"):
                LOGGER.warning(
                    "Edited message channel is unavailable (channel_id=%s, message_id=%s)",
                    payload.channel_id,
                    payload.message_id,
                )
                return
            message = await channel.fetch_message(payload.message_id)
        except discord.NotFound:
            LOGGER.info(
                "Edited message no longer exists (channel_id=%s, message_id=%s)",
                payload.channel_id,
                payload.message_id,
            )
            return
        except (discord.Forbidden, discord.HTTPException) as exc:
            LOGGER.warning(
                "Could not fetch edited message (channel_id=%s, message_id=%s, error=%s)",
                payload.channel_id,
                payload.message_id,
                type(exc).__name__,
            )
            return

        await self._process_message(message)

    async def _process_message(self, message: discord.Message) -> None:
        guild_id = message.guild.id if message.guild else None
        parent_channel_id = getattr(message.channel, "parent_id", None)
        context = MessageContext(
            guild_id=guild_id,
            channel_id=message.channel.id,
            parent_channel_id=parent_channel_id,
            author_is_bot=message.author.bot,
            webhook_id=message.webhook_id,
            has_text=bool(message.content and message.content.strip()),
        )
        if not should_monitor_message(
            context,
            self._config.guild_ids,
            self._config.monitored_channel_ids,
            self._config.monitor_all_guilds,
        ):
            return

        reply_context = self._reply_context(message)
        recent_context = self._context.recent(
            message.channel.id,
            excluding_message_id=message.id,
        )
        self._context.remember(message.channel.id, message.id, message.content)
        result = await self._service.analyze(
            message.content,
            reply_context=reply_context,
            recent_context=recent_context,
        )
        self._messages_analyzed += 1
        self._contextual_analyses += int(bool(reply_context or recent_context))
        if not result.detected:
            return

        detections_by_category = {
            detection.category: detection for detection in result.detections
        }
        claimed_categories = self._alerts.claim_new(
            message.id, detections_by_category.keys()
        )
        if not claimed_categories:
            return

        claimed_detections = [
            detections_by_category[category] for category in claimed_categories
        ]
        try:
            await self._send_alert(message, claimed_detections)
        except (discord.Forbidden, discord.NotFound, discord.HTTPException) as exc:
            self._alerts.release(message.id, claimed_categories)
            LOGGER.warning(
                "Could not send moderation alert (channel_id=%s, message_id=%s, error=%s)",
                message.channel.id,
                message.id,
                type(exc).__name__,
            )
            return
        except Exception:
            self._alerts.release(message.id, claimed_categories)
            LOGGER.exception(
                "Unexpected alert failure (channel_id=%s, message_id=%s)",
                message.channel.id,
                message.id,
            )
            return

        LOGGER.info(
            "Moderation alert sent (channel_id=%s, message_id=%s, categories=%s)",
            message.channel.id,
            message.id,
            ",".join(claimed_categories),
        )
        self._text_alerts_sent += 1

    async def _send_alert(
        self, message: discord.Message, detections: list[CategoryDetection]
    ) -> None:
        guild_id = message.guild.id
        alert_channel_id = self._alert_channels.get(guild_id)
        if alert_channel_id is None:
            alert_channel_id = self._config.alert_channel_for(guild_id)
        allowed_mentions = discord.AllowedMentions.none()
        author_name = getattr(message.author, "display_name", None) or getattr(
            message.author, "name", "不明なユーザー"
        )
        if alert_channel_id is None:
            await message.reply(
                build_alert_text(
                    detections,
                    author_name,
                    source_excerpt=message.content,
                ),
                mention_author=False,
                allowed_mentions=allowed_mentions,
            )
            return

        alert_channel = await self._resolve_channel(alert_channel_id)
        if alert_channel is None or not hasattr(alert_channel, "send"):
            raise RuntimeError("configured alert channel is unavailable")
        alert_guild = getattr(alert_channel, "guild", None)
        if alert_guild is None or alert_guild.id != guild_id:
            raise RuntimeError("configured alert channel is outside the source guild")
        await alert_channel.send(
            build_alert_text(
                detections,
                author_name,
                jump_url=message.jump_url,
                source_excerpt=message.content,
            ),
            allowed_mentions=allowed_mentions,
        )

    async def _send_voice_alert(
        self,
        guild_id: int,
        user_id: int,
        detections: list[CategoryDetection],
        transcript: str = "",
    ) -> None:
        alert_channel_id = self._alert_channels.get(guild_id)
        if alert_channel_id is None:
            alert_channel_id = self._config.voice_alert_channel_for(guild_id)
        if alert_channel_id is not None:
            alert_channel = await self._resolve_channel(alert_channel_id)
        else:
            alert_channel = self._find_automatic_alert_channel(guild_id)
        if alert_channel is None or not hasattr(alert_channel, "send"):
            raise RuntimeError("configured VC alert channel is unavailable")
        alert_guild = getattr(alert_channel, "guild", None)
        if alert_guild is None or alert_guild.id != guild_id:
            raise RuntimeError("configured VC alert channel is outside the source guild")
        guild = self.get_guild(guild_id)
        member = guild.get_member(user_id) if guild is not None else None
        speaker_name = (
            getattr(member, "display_name", None)
            or getattr(member, "name", None)
            or f"ユーザーID {user_id}"
        )
        await alert_channel.send(
            build_voice_alert_text(
                detections,
                speaker_name,
                source_excerpt=transcript,
            ),
            allowed_mentions=discord.AllowedMentions.none(),
        )
        LOGGER.info(
            "VC moderation alert sent (guild_id=%s, categories=%s)",
            guild_id,
            ",".join(item.category for item in detections),
        )

    async def _send_voice_notice(
        self,
        guild_id: int,
        channel_id: Optional[int],
        text: str,
    ) -> None:
        candidates = []
        if channel_id is not None:
            channel = await self._resolve_channel(channel_id)
            if channel is not None:
                candidates.append(channel)
        configured_id = self._alert_channels.get(guild_id)
        if configured_id is None:
            configured_id = self._config.voice_alert_channel_for(guild_id)
        if configured_id is not None and configured_id != channel_id:
            configured = await self._resolve_channel(configured_id)
            if configured is not None:
                candidates.append(configured)
        automatic = self._find_automatic_alert_channel(guild_id)
        if automatic is not None and automatic not in candidates:
            candidates.append(automatic)

        for channel in candidates:
            channel_guild = getattr(channel, "guild", None)
            if channel_guild is None or channel_guild.id != guild_id:
                continue
            if not hasattr(channel, "send"):
                continue
            try:
                await channel.send(
                    text,
                    allowed_mentions=discord.AllowedMentions.none(),
                )
                return
            except (discord.Forbidden, discord.NotFound, discord.HTTPException):
                continue
        raise RuntimeError("no sendable channel for VC notice")

    def _register_application_commands(self) -> None:
        @app_commands.command(
            name="vc_join",
            description="あなたがいるVCへBotを参加させ、監視を開始します",
        )
        @app_commands.guild_only()
        @app_commands.default_permissions(manage_guild=True)
        async def vc_join(interaction: discord.Interaction) -> None:
            await self._command_vc_join(interaction)

        @app_commands.command(
            name="vc_leave",
            description="BotをVCから退出させ、自動参加を一時停止します",
        )
        @app_commands.guild_only()
        @app_commands.default_permissions(manage_guild=True)
        async def vc_leave(interaction: discord.Interaction) -> None:
            await self._command_vc_leave(interaction)

        @app_commands.command(
            name="vc_auto",
            description="このサーバーのVC自動参加を有効・無効にします",
        )
        @app_commands.describe(enabled="自動参加を有効にする場合はオン")
        @app_commands.guild_only()
        @app_commands.default_permissions(manage_guild=True)
        async def vc_auto(
            interaction: discord.Interaction, enabled: bool
        ) -> None:
            await self._command_vc_auto(interaction, enabled)

        @app_commands.command(
            name="vc_status",
            description="このサーバーのVC監視状態を表示します",
        )
        @app_commands.guild_only()
        async def vc_status(interaction: discord.Interaction) -> None:
            await self._command_vc_status(interaction)

        @app_commands.command(
            name="bot_status",
            description="Botのテキスト・VC監視状態を表示します",
        )
        @app_commands.guild_only()
        async def bot_status(interaction: discord.Interaction) -> None:
            await self._command_bot_status(interaction)

        @app_commands.command(
            name="permissions",
            description="このチャンネルでBotが利用できる権限を確認します",
        )
        @app_commands.guild_only()
        async def permissions(interaction: discord.Interaction) -> None:
            await self._command_permissions(interaction)

        @app_commands.command(name="ping", description="Botの応答速度を確認します")
        async def ping(interaction: discord.Interaction) -> None:
            await interaction.response.send_message(
                f"🏓 応答速度: {round(self.latency * 1000)}ms",
                ephemeral=True,
            )

        @app_commands.command(
            name="moderation_help",
            description="モデレーションBotの操作コマンド一覧を表示します",
        )
        async def moderation_help(interaction: discord.Interaction) -> None:
            await interaction.response.send_message(
                "**利用できるコマンド**\n"
                "`/vc_join` 今いるVCへ参加\n"
                "`/vc_leave` VCから退出して自動参加を一時停止\n"
                "`/vc_auto` VC自動参加のオン・オフ\n"
                "`/vc_status` VC監視状態\n"
                "`/bot_status` Bot全体の状態\n"
                "`/permissions` この場所でのBot実効権限\n"
                "`/alert_channel` テキスト・VC通知先を設定\n"
                "`/alert_channel_reset` 通知先設定を初期値へ戻す\n"
                "`/alert_channel_status` 現在の通知先を確認\n"
                "`/ping` 応答速度\n\n"
                "VCと通知先の変更コマンドには「サーバーを管理」権限が必要です。",
                ephemeral=True,
            )

        @app_commands.command(
            name="alert_channel",
            description="このサーバーのテキスト・VC検知通知先を設定します",
        )
        @app_commands.describe(channel="検知通知を送るテキストチャンネル")
        @app_commands.guild_only()
        @app_commands.default_permissions(manage_guild=True)
        async def alert_channel(
            interaction: discord.Interaction,
            channel: discord.TextChannel,
        ) -> None:
            await self._command_alert_channel(interaction, channel)

        @app_commands.command(
            name="alert_channel_reset",
            description="通知先を環境設定または自動選択へ戻します",
        )
        @app_commands.guild_only()
        @app_commands.default_permissions(manage_guild=True)
        async def alert_channel_reset(interaction: discord.Interaction) -> None:
            await self._command_alert_channel_reset(interaction)

        @app_commands.command(
            name="alert_channel_status",
            description="このサーバーの現在の検知通知先を表示します",
        )
        @app_commands.guild_only()
        @app_commands.default_permissions(manage_guild=True)
        async def alert_channel_status(interaction: discord.Interaction) -> None:
            await self._command_alert_channel_status(interaction)

        for command in (
            vc_join,
            vc_leave,
            vc_auto,
            vc_status,
            bot_status,
            permissions,
            alert_channel,
            alert_channel_reset,
            alert_channel_status,
            ping,
            moderation_help,
        ):
            self.tree.add_command(command)

    async def _sync_commands_for_guild(self, guild: discord.Guild) -> None:
        if guild.id in self._synced_command_guilds:
            return
        try:
            guild_object = discord.Object(id=guild.id)
            self.tree.copy_global_to(guild=guild_object)
            await self.tree.sync(guild=guild_object)
            self._synced_command_guilds.add(guild.id)
            LOGGER.info("Application commands synced (guild_id=%s)", guild.id)
        except (discord.Forbidden, discord.HTTPException) as exc:
            LOGGER.warning(
                "Could not sync application commands (guild_id=%s, error=%s)",
                guild.id,
                type(exc).__name__,
            )

    @staticmethod
    def _can_manage_guild(interaction: discord.Interaction) -> bool:
        permissions = getattr(interaction, "permissions", None)
        return bool(permissions and permissions.manage_guild)

    async def _require_guild_manager(
        self, interaction: discord.Interaction
    ) -> bool:
        if interaction.guild is not None and self._can_manage_guild(interaction):
            return True
        await interaction.response.send_message(
            "この操作には「サーバーを管理」権限が必要です。",
            ephemeral=True,
        )
        return False

    async def _command_alert_channel(
        self,
        interaction: discord.Interaction,
        channel: discord.TextChannel,
    ) -> None:
        if not await self._require_guild_manager(interaction):
            return
        guild = interaction.guild
        channel_guild = getattr(channel, "guild", None)
        if channel_guild is None or channel_guild.id != guild.id:
            await interaction.response.send_message(
                "同じサーバーのテキストチャンネルを選んでください。",
                ephemeral=True,
            )
            return
        member = guild.me
        permissions = channel.permissions_for(member) if member is not None else None
        if not permissions or not (
            permissions.view_channel and permissions.send_messages
        ):
            await interaction.response.send_message(
                "そのチャンネルではBotに「チャンネルを見る」と"
                "「メッセージを送信」権限が必要です。",
                ephemeral=True,
            )
            return
        try:
            self._alert_channels.set(guild.id, channel.id)
        except OSError as exc:
            LOGGER.warning(
                "Could not persist alert destination (guild_id=%s, error=%s)",
                guild.id,
                type(exc).__name__,
            )
            await interaction.response.send_message(
                "通知先を保存できませんでした。少し待ってから再実行してください。",
                ephemeral=True,
            )
            return
        LOGGER.info(
            "Alert destination updated (guild_id=%s, channel_id=%s)",
            guild.id,
            channel.id,
        )
        await interaction.response.send_message(
            f"✅ テキスト・VCの検知通知先を {channel.mention} に設定しました。",
            ephemeral=True,
            allowed_mentions=discord.AllowedMentions.none(),
        )

    async def _command_alert_channel_reset(
        self, interaction: discord.Interaction
    ) -> None:
        if not await self._require_guild_manager(interaction):
            return
        try:
            changed = self._alert_channels.clear(interaction.guild.id)
        except OSError as exc:
            LOGGER.warning(
                "Could not reset alert destination (guild_id=%s, error=%s)",
                interaction.guild.id,
                type(exc).__name__,
            )
            await interaction.response.send_message(
                "通知先設定を解除できませんでした。少し待ってから再実行してください。",
                ephemeral=True,
            )
            return
        LOGGER.info(
            "Alert destination reset (guild_id=%s, changed=%s)",
            interaction.guild.id,
            changed,
        )
        await interaction.response.send_message(
            "✅ コマンドで指定した通知先を解除しました。"
            "環境設定または自動選択へ戻ります。",
            ephemeral=True,
        )

    async def _command_alert_channel_status(
        self, interaction: discord.Interaction
    ) -> None:
        if not await self._require_guild_manager(interaction):
            return
        guild_id = interaction.guild.id
        override = self._alert_channels.get(guild_id)
        if override is not None:
            status = f"コマンド設定：<#{override}>"
        else:
            text_channel = self._config.alert_channel_for(guild_id)
            voice_channel = self._config.voice_alert_channel_for(guild_id)
            if text_channel is None and voice_channel is None:
                status = "未指定（テキストは元投稿へ返信、VCは送信可能先を自動選択）"
            elif text_channel == voice_channel:
                status = f"環境設定：<#{text_channel}>"
            else:
                text_status = (
                    f"<#{text_channel}>"
                    if text_channel is not None
                    else "元投稿へ返信"
                )
                voice_status = (
                    f"<#{voice_channel}>"
                    if voice_channel is not None
                    else "自動選択"
                )
                status = f"環境設定：テキスト={text_status}、VC={voice_status}"
        await interaction.response.send_message(
            f"**現在の検知通知先**\n{status}",
            ephemeral=True,
            allowed_mentions=discord.AllowedMentions.none(),
        )

    async def _command_vc_join(self, interaction: discord.Interaction) -> None:
        if not await self._require_guild_manager(interaction):
            return
        voice_state = getattr(interaction.user, "voice", None)
        channel = getattr(voice_state, "channel", None)
        if channel is None:
            await interaction.response.send_message(
                "先に参加したいVCへ入ってから実行してください。",
                ephemeral=True,
            )
            return
        try:
            await self._voice.join_channel(interaction.guild.id, channel.id)
        except Exception as exc:
            LOGGER.warning(
                "VC join command failed (guild_id=%s, error=%s)",
                interaction.guild.id,
                type(exc).__name__,
            )
            await interaction.response.send_message(
                "VCへ参加できませんでした。BotのConnect権限を確認してください。",
                ephemeral=True,
            )
            return
        await interaction.response.send_message(
            f"✅ VC「{channel.name}」へ参加し、監視を開始しました。",
            ephemeral=True,
        )

    async def _command_vc_leave(self, interaction: discord.Interaction) -> None:
        if not await self._require_guild_manager(interaction):
            return
        try:
            await self._voice.leave_guild(interaction.guild.id)
        except Exception as exc:
            LOGGER.warning(
                "VC leave command failed (guild_id=%s, error=%s)",
                interaction.guild.id,
                type(exc).__name__,
            )
            await interaction.response.send_message(
                "VCから退出できませんでした。少し待ってから再実行してください。",
                ephemeral=True,
            )
            return
        await interaction.response.send_message(
            "✅ VCから退出し、このサーバーの自動参加を一時停止しました。"
            "再開は `/vc_auto enabled:True` です。",
            ephemeral=True,
        )

    async def _command_vc_auto(
        self, interaction: discord.Interaction, enabled: bool
    ) -> None:
        if not await self._require_guild_manager(interaction):
            return
        try:
            await self._voice.set_auto_join_for_guild(interaction.guild.id, enabled)
        except Exception as exc:
            LOGGER.warning(
                "VC auto command failed (guild_id=%s, error=%s)",
                interaction.guild.id,
                type(exc).__name__,
            )
            await interaction.response.send_message(
                "VC自動参加を変更できませんでした。",
                ephemeral=True,
            )
            return
        state = "有効" if enabled else "一時停止"
        channel = self._voice.current_channel(interaction.guild.id)
        if enabled and channel is None:
            detail = " 現在利用者がいるVCがないため、参加者が入るまで待機します。"
        elif enabled:
            detail = f" VC「{channel.name}」を監視中です。"
        else:
            detail = " 現在のVCからも退出しました。"
        LOGGER.info(
            "VC automatic monitoring changed (guild_id=%s, enabled=%s)",
            interaction.guild.id,
            enabled,
        )
        await interaction.response.send_message(
            f"✅ このサーバーのVC自動参加を{state}にしました。{detail}",
            ephemeral=True,
        )

    async def _command_vc_status(self, interaction: discord.Interaction) -> None:
        guild_id = interaction.guild.id
        channel = self._voice.current_channel(guild_id)
        channel_text = f"VC「{channel.name}」を監視中" if channel else "VC未参加"
        auto_text = (
            "有効" if self._voice.is_auto_join_enabled(guild_id) else "一時停止"
        )
        await interaction.response.send_message(
            f"🎙️ {channel_text}\n自動参加: {auto_text}\n"
            f"{self._voice.status_summary(guild_id)}\n"
            "音声と文字起こしはBotのファイルへ保存しません。",
            ephemeral=True,
        )

    async def _command_bot_status(self, interaction: discord.Interaction) -> None:
        scope = "参加中の全サーバー" if self._config.monitor_all_guilds else "指定サーバー"
        voice_state = "有効" if self._config.voice_enabled else "無効"
        backend = getattr(self._service, "backend_name", "ローカルルール")
        health = getattr(self._service, "health_summary", "ローカル動作中")
        uptime = self._format_duration(time.monotonic() - self._started_at)
        connected_voice = sum(
            1 for guild in self.guilds if self._voice.current_channel(guild.id)
        )
        reconnect = (
            "なし"
            if self._last_resumed_at is None
            else f"{self._format_duration(time.monotonic() - self._last_resumed_at)}前"
        )
        await interaction.response.send_message(
            "✅ Botはオンラインです。\n"
            f"稼働時間: {uptime}\n"
            f"テキスト監視: {scope}の閲覧可能な全チャンネル\n"
            f"判定済み: {self._messages_analyzed}件（会話文脈使用: {self._contextual_analyses}件）\n"
            f"テキスト通知: {self._text_alerts_sent}件\n"
            f"メモリ内文脈: {self._context.message_count}件・最大"
            f"{self._config.context_message_count}件/チャンネル・"
            f"{self._config.context_ttl_seconds}秒で消去\n"
            f"VC監視: {voice_state}・接続中{connected_voice}サーバー\n"
            f"判定・音声認識: {backend}\n"
            f"API状態: {health}\n"
            f"Discord再接続: {reconnect}\n"
            "本文・音声・文字起こしはBotのファイルへ保存しません。",
            ephemeral=True,
        )

    async def _command_permissions(self, interaction: discord.Interaction) -> None:
        guild = interaction.guild
        channel = interaction.channel
        member = guild.me if guild is not None else None
        if member is None or channel is None or not hasattr(channel, "permissions_for"):
            await interaction.response.send_message(
                "この場所の権限を確認できませんでした。", ephemeral=True
            )
            return
        permissions = channel.permissions_for(member)
        voice_channels = (*guild.voice_channels, *guild.stage_channels)
        voice_connectable = sum(
            1
            for voice_channel in voice_channels
            if voice_channel.permissions_for(member).view_channel
            and voice_channel.permissions_for(member).connect
        )
        checks = (
            ("チャンネルを見る", permissions.view_channel),
            ("メッセージを送信", permissions.send_messages),
            ("履歴を読む", permissions.read_message_history),
            ("スレッドへ送信", permissions.send_messages_in_threads),
        )
        report = "\n".join(
            f"{'✅' if enabled else '❌'} {label}" for label, enabled in checks
        )
        await interaction.response.send_message(
            f"**このチャンネルでのBot実効権限**\n{report}\n"
            f"{'✅' if voice_connectable else '❌'} VCへ接続: "
            f"{voice_connectable}/{len(voice_channels)}チャンネル\n"
            "✅ Message Content Intent: 有効\n"
            "✅ スラッシュコマンド: 同期済み",
            ephemeral=True,
        )

    @staticmethod
    def _reply_context(message: discord.Message) -> Optional[str]:
        reference = getattr(message, "reference", None)
        resolved = getattr(reference, "resolved", None)
        content = getattr(resolved, "content", None)
        if isinstance(content, str) and content.strip():
            return content
        return None

    def _audit_guild_permissions(self, guild: discord.Guild) -> None:
        member = guild.me
        if member is None:
            LOGGER.warning("Could not audit Bot permissions (guild_id=%s)", guild.id)
            return
        text_visible = text_sendable = text_history = 0
        for channel in guild.text_channels:
            permissions = channel.permissions_for(member)
            text_visible += int(permissions.view_channel)
            text_sendable += int(
                permissions.view_channel and permissions.send_messages
            )
            text_history += int(
                permissions.view_channel and permissions.read_message_history
            )
        voice_connectable = sum(
            1
            for channel in (*guild.voice_channels, *guild.stage_channels)
            if channel.permissions_for(member).view_channel
            and channel.permissions_for(member).connect
        )
        LOGGER.info(
            "Bot permission audit (guild_id=%s, text_visible=%s, text_sendable=%s, "
            "text_history=%s, voice_connectable=%s, commands_synced=%s)",
            guild.id,
            text_visible,
            text_sendable,
            text_history,
            voice_connectable,
            guild.id in self._synced_command_guilds,
        )

    def _find_automatic_alert_channel(
        self, guild_id: int
    ) -> Optional[discord.TextChannel]:
        guild = self.get_guild(guild_id)
        if guild is None or guild.me is None:
            return None
        candidates = []
        if guild.system_channel is not None:
            candidates.append(guild.system_channel)
        candidates.extend(
            channel
            for channel in guild.text_channels
            if channel not in candidates
        )
        for channel in candidates:
            permissions = channel.permissions_for(guild.me)
            if permissions.view_channel and permissions.send_messages:
                return channel
        return None

    async def _resolve_channel(self, channel_id: int) -> Optional[discord.abc.GuildChannel]:
        channel = self.get_channel(channel_id)
        if channel is not None:
            return channel
        try:
            return await self.fetch_channel(channel_id)
        except (discord.NotFound, discord.Forbidden, discord.HTTPException) as exc:
            LOGGER.warning(
                "Could not resolve channel (channel_id=%s, error=%s)",
                channel_id,
                type(exc).__name__,
            )
            return None

    async def close(self) -> None:
        self._context.clear()
        await self._voice.close()
        await self._service.close()
        await super().close()

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total = max(0, int(seconds))
        days, remainder = divmod(total, 86_400)
        hours, remainder = divmod(remainder, 3_600)
        minutes, secs = divmod(remainder, 60)
        if days:
            return f"{days}日{hours}時間"
        if hours:
            return f"{hours}時間{minutes}分"
        if minutes:
            return f"{minutes}分{secs}秒"
        return f"{secs}秒"


def create_client(config: BotConfig, service: ModerationService) -> ModerationClient:
    return ModerationClient(config, service)
