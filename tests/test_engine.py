from pathlib import Path
import unittest

from discord_moderation_bot.engine import ModerationEngine, normalize_text


RULES_PATH = Path(__file__).resolve().parent.parent / "config" / "rules.json"


class NormalizeTextTests(unittest.TestCase):
    def test_normalizes_width_case_controls_and_whitespace(self) -> None:
        self.assertEqual(normalize_text("  Ａ\u200bＢ\n  Ｃ  "), "ab c")


class ModerationEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = ModerationEngine.from_json(RULES_PATH)

    def categories_for(self, text: str):
        result = self.engine.analyze(text)
        return {detection.category for detection in result.detections}

    def test_explicit_slur_is_detected_after_obfuscation(self) -> None:
        result = self.engine.analyze("土\u200b人")
        self.assertTrue(result.detected)
        self.assertEqual(self.categories_for("土\u200b人"), {"discrimination"})
        self.assertEqual(self.categories_for("ガ・イ・ジ"), {"discrimination"})

    def test_attribute_plus_exclusion_is_discrimination(self) -> None:
        self.assertEqual(self.categories_for("外国人は出ていけ"), {"discrimination"})

    def test_targeted_ridicule_is_cynicism(self) -> None:
        self.assertEqual(self.categories_for("お前、必死で草"), {"cynicism"})

    def test_implicit_target_ridicule_is_cynicism(self) -> None:
        self.assertEqual(self.categories_for("そんなこともできないの"), {"cynicism"})

    def test_uow_laugh_variants_are_cynicism(self) -> None:
        for text in ("うおw", "うおww", "うおｗ", "う お W"):
            with self.subTest(text=text):
                self.assertEqual(self.categories_for(text), {"cynicism"})

    def test_uo_without_laughter_is_also_cynicism(self) -> None:
        for text in ("うお", "う お", "え、うお。"):
            with self.subTest(text=text):
                self.assertEqual(self.categories_for(text), {"cynicism"})
        self.assertFalse(self.engine.analyze("うお座の話").detected)

    def test_dowa_variants_are_cynicism_but_dwarf_is_not(self) -> None:
        for text in ("どわー", "どわ〜", "ドワーーw"):
            with self.subTest(text=text):
                self.assertEqual(self.categories_for(text), {"cynicism"})
        self.assertFalse(self.engine.analyze("ドワーフの冒険").detected)

    def test_gendered_emasculation_is_discrimination(self) -> None:
        for text in (
            "お前は女々しい",
            "女々しくて情けない",
            "男らしくない",
            "男のくせに泣くな",
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    self.categories_for(text),
                    {"discrimination", "cynicism"},
                )

    def test_meu_is_cynicism_only_at_the_end_of_a_message(self) -> None:
        for text in ("了解めう", "今日は行くめう！", "むりめうw"):
            with self.subTest(text=text):
                self.assertEqual(self.categories_for(text), {"cynicism"})
        self.assertFalse(self.engine.analyze("めうという語尾について話す").detected)

    def test_kuiya_is_a_community_cynicism_term(self) -> None:
        for text in ("クイヤ", "それクイヤだ", "くいや"):
            with self.subTest(text=text):
                self.assertEqual(self.categories_for(text), {"cynicism"})

    def test_sexual_jokes_and_explicit_terms_have_their_own_category(self) -> None:
        for text in ("下ネタやめて", "セックスの話", "おっぱい", "エッチな冗談"):
            with self.subTest(text=text):
                self.assertEqual(self.categories_for(text), {"sexual_content"})
        for neutral in ("エッチング加工", "処女航海", "ちんちん電車"):
            with self.subTest(text=neutral):
                self.assertFalse(self.engine.analyze(neutral).detected)

    def test_adhd_is_always_a_separate_sensitive_term_category(self) -> None:
        for text in ("ADHD", "私はADHDです", "adhdについて"):
            with self.subTest(text=text):
                self.assertEqual(self.categories_for(text), {"sensitive_term"})
        self.assertFalse(self.engine.analyze("ADHDersという英単語").detected)

    def test_drug_related_terms_have_their_own_category(self) -> None:
        for text in (
            "大麻の話",
            "マリファナを買いたい",
            "覚醒剤",
            "コカインを売る",
            "MDMA",
            "LSDについて",
            "オーバードーズした",
            "ODしよう",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.categories_for(text), {"drug_content"})
        for neutral in ("病院で薬を処方された", "ドラッグストアへ行く", "OD缶を買う"):
            with self.subTest(text=neutral):
                self.assertFalse(self.engine.analyze(neutral).detected)

    def test_complex_discrimination_phrasings_are_detected(self) -> None:
        examples = (
            "移民はみんな信用できない",
            "女性は生まれつき能力が低い",
            "障害は甘え",
            "ゲイは病気だから治すべき",
            "難民を店に入れるな",
            "採用するな、外国人は信用できない",
            "生活保護受給者は税金泥棒だ",
        )
        for text in examples:
            with self.subTest(text=text):
                self.assertEqual(self.categories_for(text), {"discrimination"})

    def test_expanded_japanese_and_english_slurs_are_detected(self) -> None:
        for text in (
            "ちゃんころ",
            "毛唐",
            "知恵遅れ",
            "nigger",
            "faggot",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.categories_for(text), {"discrimination"})

    def test_more_protected_groups_with_hostility_are_detected(self) -> None:
        for text in (
            "クルド人は国へ帰れ",
            "トランスジェンダーを雇うな",
            "認知症の人は社会のお荷物だ",
            "母子家庭なんて全員信用できない",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.categories_for(text), {"discrimination"})

    def test_more_direct_abuse_and_mockery_are_detected(self) -> None:
        for text in (
            "死ね",
            "黙れ",
            "帰れ",
            "ざまあ",
            "涙拭けよ",
            "you are useless",
            "お前は役立たず",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.categories_for(text), {"cynicism"})

    def test_high_sensitivity_detects_standalone_abuse_and_dehumanization(self) -> None:
        cases = {
            "バカ": {"cynicism"},
            "キモい": {"cynicism"},
            "害虫": {"discrimination"},
            "お前w": {"cynicism"},
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(self.categories_for(text), expected)

    def test_high_sensitivity_keeps_basic_neutral_terms_safe(self) -> None:
        for text in ("草", "女性", "魚料理"):
            with self.subTest(text=text):
                self.assertFalse(self.engine.analyze(text).detected)

    def test_more_sexual_and_drug_terms_are_detected(self) -> None:
        cases = {
            "中出し": {"sexual_content"},
            "ポルノ": {"sexual_content"},
            "porn": {"sexual_content"},
            "DMT": {"drug_content"},
            "阿片": {"drug_content"},
            "葉っぱを吸う": {"drug_content"},
            "アイスをキメる": {"drug_content"},
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(self.categories_for(text), expected)

    def test_expanded_obfuscation_separators_are_removed(self) -> None:
        for text in ("ガ/イ/ジ", "死｜ね", "f-a-g-g-o-t"):
            with self.subTest(text=text):
                self.assertTrue(self.engine.analyze(text).detected)

    def test_ambiguous_new_terms_need_hostile_context(self) -> None:
        for text in (
            "庭の草を刈る",
            "アイスを食べる",
            "仕事のスピードを上げる",
            "母子家庭への支援を増やす",
            "認知症について学ぶ",
            "you are helpful",
        ):
            with self.subTest(text=text):
                self.assertFalse(self.engine.analyze(text).detected)

    def test_combined_message_returns_both_categories(self) -> None:
        self.assertEqual(
            self.categories_for("お前みたいな外国人は出ていけ。必死で草"),
            {"discrimination", "cynicism"},
        )

    def test_laughter_alone_is_not_detected(self) -> None:
        self.assertFalse(self.engine.analyze("草").detected)

    def test_neutral_attribute_reference_is_not_detected(self) -> None:
        for text in (
            "外国人向けの案内を作ります",
            "女性の管理職が増えた",
            "うつ病への偏見をなくそう",
            "難民の生活支援について話し合う",
        ):
            with self.subTest(text=text):
                self.assertFalse(self.engine.analyze(text).detected)

    def test_general_joke_is_not_detected(self) -> None:
        self.assertFalse(self.engine.analyze("この猫の動画おもしろくて笑った").detected)

    def test_critical_quote_is_suppressed(self) -> None:
        self.assertFalse(
            self.engine.analyze("「外国人は出ていけ」という発言は差別でよくない").detected
        )

    def test_result_does_not_retain_source_text(self) -> None:
        source = "外国人は出ていけ"
        result = self.engine.analyze(source)
        self.assertNotIn(source, repr(result))
        self.assertNotIn("text", vars(result))


if __name__ == "__main__":
    unittest.main()
