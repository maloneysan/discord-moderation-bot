import unittest

from discord_moderation_bot.state import AlertRegistry


class AlertRegistryTests(unittest.TestCase):
    def test_same_message_and_category_is_claimed_once(self) -> None:
        registry = AlertRegistry()
        self.assertEqual(registry.claim_new(1, ["discrimination"]), ("discrimination",))
        self.assertEqual(registry.claim_new(1, ["discrimination"]), ())

    def test_edit_can_claim_a_new_category(self) -> None:
        registry = AlertRegistry()
        registry.claim_new(1, ["discrimination"])
        self.assertEqual(
            registry.claim_new(1, ["discrimination", "cynicism"]), ("cynicism",)
        )

    def test_failed_alert_can_be_released_and_retried(self) -> None:
        registry = AlertRegistry()
        registry.claim_new(1, ["cynicism"])
        registry.release(1, ["cynicism"])
        self.assertEqual(registry.claim_new(1, ["cynicism"]), ("cynicism",))

    def test_registry_is_bounded(self) -> None:
        registry = AlertRegistry(max_entries=2)
        registry.claim_new(1, ["cynicism"])
        registry.claim_new(2, ["cynicism"])
        registry.claim_new(3, ["cynicism"])
        self.assertEqual(registry.claim_new(1, ["cynicism"]), ("cynicism",))


if __name__ == "__main__":
    unittest.main()
