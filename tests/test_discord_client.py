from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock

import discord

from discord_moderation_bot.bot import create_client
from discord_moderation_bot.config import BotConfig
from discord_moderation_bot.engine import ModerationEngine
from discord_moderation_bot.models import CategoryDetection, DetectionResult
from discord_moderation_bot.service import LocalModerationService


RULES_PATH = Path(__file__).resolve().parent.parent / "config" / "rules.json"


def detection(category="discrimination", label="差別表現"):
    reason = (
        "属性を理由に排除する内容です。"
        if category == "discrimination"
        else "相手を嘲笑する内容です。"
    )
    return CategoryDetection(category, label, 90, ("test.rule",), reason)


class DiscordClientTests(unittest.IsolatedAsyncioTestCase):
    def make_client(self, alert_channel_ids=None, monitor_all_guilds=False):
        config = BotConfig(
            token="not-a-real-token",
            guild_ids=frozenset({100, 101}),
            monitored_channel_ids=frozenset(),
            alert_channel_ids=alert_channel_ids or {},
            rules_path=RULES_PATH,
            monitor_all_guilds=monitor_all_guilds,
        )
        return create_client(
            config, LocalModerationService(ModerationEngine.from_json(RULES_PATH))
        )

    @staticmethod
    def make_message(guild_id=100):
        return SimpleNamespace(
            id=400,
            guild=SimpleNamespace(id=guild_id),
            channel=SimpleNamespace(id=200, parent_id=None),
            author=SimpleNamespace(
                bot=False, display_name="テスト話者", name="test-user"
            ),
            webhook_id=None,
            content="外国人は出ていけ",
            jump_url=f"https://discord.com/channels/{guild_id}/200/400",
            reply=AsyncMock(),
        )

    async def test_reply_disables_author_and_all_mentions(self) -> None:
        client = self.make_client()
        message = self.make_message()

        await client._send_alert(message, [detection()])

        message.reply.assert_awaited_once()
        _, kwargs = message.reply.await_args
        self.assertFalse(kwargs["mention_author"])
        self.assertEqual(
            kwargs["allowed_mentions"].to_dict(),
            discord.AllowedMentions.none().to_dict(),
        )
        self.assertIn("発言者：テスト話者", message.reply.await_args.args[0])
        self.assertIn("問題だった点：", message.reply.await_args.args[0])
        self.assertIn("・差別表現：", message.reply.await_args.args[0])
        self.assertIn("該当発言：", message.reply.await_args.args[0])
        self.assertIn(message.content, message.reply.await_args.args[0])

    async def test_processing_alerts_once_then_only_for_new_edit_category(self) -> None:
        client = self.make_client()
        client._send_alert = AsyncMock()
        message = self.make_message()

        await client._process_message(message)
        await client._process_message(message)
        message.content = "お前みたいな外国人は出ていけ。必死で草"
        await client._process_message(message)

        self.assertEqual(client._send_alert.await_count, 2)
        first_args = client._send_alert.await_args_list[0].args
        second_args = client._send_alert.await_args_list[1].args
        self.assertEqual([item.label for item in first_args[1]], ["差別表現"])
        self.assertEqual([item.label for item in second_args[1]], ["冷笑"])

    async def test_previous_messages_are_used_as_ephemeral_context(self) -> None:
        client = self.make_client()
        client._service.analyze = AsyncMock(return_value=DetectionResult.empty())
        first = self.make_message()
        first.id = 401
        first.content = "前の会話"
        second = self.make_message()
        second.id = 402
        second.content = "現在の発言"

        await client._process_message(first)
        await client._process_message(second)

        second_call = client._service.analyze.await_args_list[1]
        self.assertEqual(second_call.kwargs["recent_context"], ("前の会話",))
        self.assertNotIn("テスト話者", second_call.kwargs["recent_context"])

    async def test_second_configured_guild_is_processed(self) -> None:
        client = self.make_client()
        client._send_alert = AsyncMock()
        await client._process_message(self.make_message(guild_id=101))
        client._send_alert.assert_awaited_once()

    async def test_unconfigured_guild_is_ignored(self) -> None:
        client = self.make_client()
        client._send_alert = AsyncMock()
        await client._process_message(self.make_message(guild_id=999))
        client._send_alert.assert_not_awaited()

    async def test_all_guild_mode_processes_another_joined_guild(self) -> None:
        client = self.make_client(monitor_all_guilds=True)
        client._send_alert = AsyncMock()
        await client._process_message(self.make_message(guild_id=999))
        client._send_alert.assert_awaited_once()

    async def test_dedicated_alert_uses_source_guild_mapping(self) -> None:
        client = self.make_client({100: 300, 101: 301})
        alert_channel = SimpleNamespace(
            guild=SimpleNamespace(id=101),
            send=AsyncMock(),
        )
        client._resolve_channel = AsyncMock(return_value=alert_channel)
        message = self.make_message(guild_id=101)

        await client._send_alert(
            message, [detection("cynicism", "冷笑")]
        )

        client._resolve_channel.assert_awaited_once_with(301)
        alert_channel.send.assert_awaited_once()
        args, kwargs = alert_channel.send.await_args
        self.assertIn(message.jump_url, args[0])
        self.assertEqual(
            kwargs["allowed_mentions"].to_dict(),
            discord.AllowedMentions.none().to_dict(),
        )

    async def test_dedicated_alert_rejects_cross_guild_channel(self) -> None:
        client = self.make_client({100: 300})
        alert_channel = SimpleNamespace(
            guild=SimpleNamespace(id=999),
            send=AsyncMock(),
        )
        client._resolve_channel = AsyncMock(return_value=alert_channel)

        with self.assertRaisesRegex(RuntimeError, "outside the source guild"):
            await client._send_alert(
                self.make_message(), [detection("cynicism", "冷笑")]
            )
        alert_channel.send.assert_not_awaited()

    async def test_voice_alert_uses_per_guild_destination_without_mentions(self) -> None:
        config = BotConfig(
            token="not-a-real-token",
            guild_ids=frozenset({100}),
            monitored_channel_ids=frozenset(),
            alert_channel_ids={},
            rules_path=RULES_PATH,
            voice_enabled=True,
            voice_channel_ids={100: 400},
            voice_alert_channel_ids={100: 300},
        )
        client = create_client(
            config, LocalModerationService(ModerationEngine.from_json(RULES_PATH))
        )
        alert_channel = SimpleNamespace(
            guild=SimpleNamespace(id=100), send=AsyncMock()
        )
        client._resolve_channel = AsyncMock(return_value=alert_channel)
        client.get_guild = Mock(
            return_value=SimpleNamespace(
                get_member=Mock(
                    return_value=SimpleNamespace(display_name="VC話者")
                )
            )
        )

        await client._send_voice_alert(
            100, 200, [detection()], "外国人は出ていけ"
        )

        args, kwargs = alert_channel.send.await_args
        self.assertIn("VC", args[0])
        self.assertNotIn("@", args[0])
        self.assertIn("発言者：VC話者", args[0])
        self.assertIn("問題だった点：", args[0])
        self.assertIn("・差別表現：", args[0])
        self.assertIn("該当発言：「外国人は出ていけ」", args[0])
        self.assertEqual(
            kwargs["allowed_mentions"].to_dict(),
            discord.AllowedMentions.none().to_dict(),
        )

    async def test_voice_alert_automatically_uses_a_sendable_text_channel(self) -> None:
        client = self.make_client(monitor_all_guilds=True)
        channel = SimpleNamespace(
            guild=SimpleNamespace(id=100),
            send=AsyncMock(),
            permissions_for=Mock(
                return_value=SimpleNamespace(
                    view_channel=True,
                    send_messages=True,
                )
            ),
        )
        guild = SimpleNamespace(
            id=100,
            me=SimpleNamespace(),
            system_channel=channel,
            text_channels=[channel],
            get_member=Mock(
                return_value=SimpleNamespace(display_name="自動VC話者")
            ),
        )
        client.get_guild = Mock(return_value=guild)

        await client._send_voice_alert(
            100, 200, [detection("cynicism", "冷笑")]
        )

        channel.send.assert_awaited_once()
        sent_text = channel.send.await_args.args[0]
        self.assertIn("VC", sent_text)
        self.assertIn("発言者：自動VC話者", sent_text)
        self.assertNotIn("@", sent_text)

    async def test_voice_monitoring_notice_disables_mentions(self) -> None:
        client = self.make_client(monitor_all_guilds=True)
        voice_channel = SimpleNamespace(
            guild=SimpleNamespace(id=100),
            send=AsyncMock(),
        )
        client._resolve_channel = AsyncMock(return_value=voice_channel)
        client._find_automatic_alert_channel = Mock(return_value=None)

        await client._send_voice_notice(100, 400, "監視を開始します")

        voice_channel.send.assert_awaited_once()
        allowed = voice_channel.send.await_args.kwargs["allowed_mentions"]
        self.assertEqual(allowed.to_dict(), discord.AllowedMentions.none().to_dict())

    async def test_leaving_monitored_vc_discards_speaker_state(self) -> None:
        config = BotConfig(
            token="not-a-real-token",
            guild_ids=frozenset({100}),
            monitored_channel_ids=frozenset(),
            alert_channel_ids={100: 300},
            rules_path=RULES_PATH,
            voice_enabled=True,
            voice_channel_ids={100: 400},
        )
        client = create_client(
            config, LocalModerationService(ModerationEngine.from_json(RULES_PATH))
        )
        client._voice.forget_speaker = Mock()
        member = SimpleNamespace(guild=SimpleNamespace(id=100), id=200)
        before = SimpleNamespace(channel=SimpleNamespace(id=400))
        after = SimpleNamespace(channel=None)

        await client.on_voice_state_update(member, before, after)

        client._voice.forget_speaker.assert_called_once_with(100, 200)

    def test_useful_application_commands_are_registered(self) -> None:
        client = self.make_client()
        names = {command.name for command in client.tree.get_commands()}
        self.assertEqual(
            names,
            {
                "vc_join",
                "vc_leave",
                "vc_auto",
                "vc_status",
                "bot_status",
                "permissions",
                "alert_channel",
                "alert_channel_reset",
                "alert_channel_status",
                "ping",
                "moderation_help",
            },
        )

    async def test_alert_channel_command_persists_sendable_same_guild_channel(self) -> None:
        client = self.make_client()
        member = SimpleNamespace()
        guild = SimpleNamespace(id=100, me=member)
        channel = SimpleNamespace(
            id=300,
            mention="<#300>",
            guild=guild,
            permissions_for=Mock(
                return_value=SimpleNamespace(view_channel=True, send_messages=True)
            ),
        )
        interaction = SimpleNamespace(
            guild=guild,
            permissions=SimpleNamespace(manage_guild=True),
            response=SimpleNamespace(send_message=AsyncMock()),
        )

        await client._command_alert_channel(interaction, channel)

        self.assertEqual(client._alert_channels.get(100), 300)
        response = interaction.response.send_message.await_args
        self.assertIn("<#300>", response.args[0])
        self.assertTrue(response.kwargs["ephemeral"])

    async def test_alert_channel_command_rejects_cross_guild_or_unsendable_channel(self) -> None:
        client = self.make_client()
        guild = SimpleNamespace(id=100, me=SimpleNamespace())
        interaction = SimpleNamespace(
            guild=guild,
            permissions=SimpleNamespace(manage_guild=True),
            response=SimpleNamespace(send_message=AsyncMock()),
        )
        cross_guild = SimpleNamespace(
            id=300,
            guild=SimpleNamespace(id=999),
        )

        await client._command_alert_channel(interaction, cross_guild)
        self.assertIsNone(client._alert_channels.get(100))

        interaction.response.send_message.reset_mock()
        blocked = SimpleNamespace(
            id=301,
            guild=guild,
            permissions_for=Mock(
                return_value=SimpleNamespace(view_channel=True, send_messages=False)
            ),
        )
        await client._command_alert_channel(interaction, blocked)
        self.assertIsNone(client._alert_channels.get(100))

    async def test_runtime_alert_channel_overrides_text_and_voice_destinations(self) -> None:
        client = self.make_client({100: 301})
        client._alert_channels.set(100, 300)
        alert_channel = SimpleNamespace(
            guild=SimpleNamespace(id=100),
            send=AsyncMock(),
        )
        client._resolve_channel = AsyncMock(return_value=alert_channel)
        client.get_guild = Mock(
            return_value=SimpleNamespace(
                get_member=Mock(return_value=SimpleNamespace(display_name="VC話者"))
            )
        )

        await client._send_alert(self.make_message(), [detection()])
        await client._send_voice_alert(100, 200, [detection()])

        self.assertEqual(
            [call.args[0] for call in client._resolve_channel.await_args_list],
            [300, 300],
        )
        self.assertEqual(alert_channel.send.await_count, 2)

    async def test_alert_channel_reset_and_status_require_manager(self) -> None:
        client = self.make_client()
        client._alert_channels.set(100, 300)
        interaction = SimpleNamespace(
            guild=SimpleNamespace(id=100),
            permissions=SimpleNamespace(manage_guild=True),
            response=SimpleNamespace(send_message=AsyncMock()),
        )

        await client._command_alert_channel_status(interaction)
        self.assertIn("<#300>", interaction.response.send_message.await_args.args[0])

        interaction.response.send_message.reset_mock()
        await client._command_alert_channel_reset(interaction)
        self.assertIsNone(client._alert_channels.get(100))

        denied = SimpleNamespace(
            guild=SimpleNamespace(id=100),
            permissions=SimpleNamespace(manage_guild=False),
            response=SimpleNamespace(send_message=AsyncMock()),
        )
        await client._command_alert_channel_status(denied)
        self.assertIn("サーバーを管理", denied.response.send_message.await_args.args[0])

    async def test_vc_join_command_joins_the_callers_channel(self) -> None:
        client = self.make_client()
        client._voice.join_channel = AsyncMock()
        channel = SimpleNamespace(id=400, name="ラウンジ")
        interaction = SimpleNamespace(
            guild=SimpleNamespace(id=100),
            user=SimpleNamespace(voice=SimpleNamespace(channel=channel)),
            permissions=SimpleNamespace(manage_guild=True),
            response=SimpleNamespace(send_message=AsyncMock()),
        )

        await client._command_vc_join(interaction)

        client._voice.join_channel.assert_awaited_once_with(100, 400)
        interaction.response.send_message.assert_awaited_once()
        self.assertTrue(interaction.response.send_message.await_args.kwargs["ephemeral"])

    async def test_vc_join_command_requires_manage_guild_permission(self) -> None:
        client = self.make_client()
        client._voice.join_channel = AsyncMock()
        interaction = SimpleNamespace(
            guild=SimpleNamespace(id=100),
            permissions=SimpleNamespace(manage_guild=False),
            response=SimpleNamespace(send_message=AsyncMock()),
        )

        await client._command_vc_join(interaction)

        client._voice.join_channel.assert_not_awaited()
        interaction.response.send_message.assert_awaited_once()

    async def test_vc_auto_command_reports_disconnection_when_disabled(self) -> None:
        client = self.make_client()
        client._voice.set_auto_join_for_guild = AsyncMock()
        client._voice.current_channel = Mock(return_value=None)
        interaction = SimpleNamespace(
            guild=SimpleNamespace(id=100),
            permissions=SimpleNamespace(manage_guild=True),
            response=SimpleNamespace(send_message=AsyncMock()),
        )

        await client._command_vc_auto(interaction, False)

        client._voice.set_auto_join_for_guild.assert_awaited_once_with(100, False)
        message = interaction.response.send_message.await_args.args[0]
        self.assertIn("現在のVCからも退出", message)

    async def test_permissions_command_reports_voice_channel_overwrites(self) -> None:
        client = self.make_client()
        member = SimpleNamespace()
        text_permissions = SimpleNamespace(
            view_channel=True,
            send_messages=True,
            read_message_history=True,
            send_messages_in_threads=False,
        )
        text_channel = SimpleNamespace(
            permissions_for=Mock(return_value=text_permissions)
        )
        connectable = SimpleNamespace(
            permissions_for=Mock(
                return_value=SimpleNamespace(view_channel=True, connect=True)
            )
        )
        blocked = SimpleNamespace(
            permissions_for=Mock(
                return_value=SimpleNamespace(view_channel=True, connect=False)
            )
        )
        guild = SimpleNamespace(
            me=member,
            voice_channels=[connectable, blocked],
            stage_channels=[],
        )
        interaction = SimpleNamespace(
            guild=guild,
            channel=text_channel,
            response=SimpleNamespace(send_message=AsyncMock()),
        )

        await client._command_permissions(interaction)

        message = interaction.response.send_message.await_args.args[0]
        self.assertIn("VCへ接続: 1/2チャンネル", message)
        self.assertIn("❌ スレッドへ送信", message)

    async def test_bot_status_reports_uptime_context_and_api_health(self) -> None:
        client = self.make_client()
        interaction = SimpleNamespace(
            response=SimpleNamespace(send_message=AsyncMock())
        )

        await client._command_bot_status(interaction)

        message = interaction.response.send_message.await_args.args[0]
        self.assertIn("稼働時間", message)
        self.assertIn("メモリ内文脈", message)
        self.assertIn("API状態", message)
        self.assertIn("Discord再接続", message)


if __name__ == "__main__":
    unittest.main()
