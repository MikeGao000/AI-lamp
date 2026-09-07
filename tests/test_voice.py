import unittest

from lamp_core.voice import INTENT_PHRASES, VoiceIntent, parse_intent


class VoiceTests(unittest.TestCase):
    def test_known_chinese_command_is_allow_listed(self):
        self.assertEqual(VoiceIntent.READ, parse_intent("小灯，继续读书").intent)
        self.assertEqual(VoiceIntent.STOP, parse_intent("请停止").intent)

    def test_unknown_voice_command_does_not_trigger_motion(self):
        self.assertEqual(VoiceIntent.NONE, parse_intent("今天天气真好").intent)

    def test_teach_commands_are_distinct(self):
        self.assertEqual(VoiceIntent.TEACH_START, parse_intent("开始示教").intent)
        self.assertEqual(VoiceIntent.TEACH_STOP, parse_intent("保存动作").intent)

    def test_specific_teach_stop_is_not_shadowed_by_normal_stop(self):
        self.assertEqual(VoiceIntent.TEACH_STOP, parse_intent("停止录制").intent)
        self.assertEqual(VoiceIntent.STOP, parse_intent("停止录制，然后停止").intent)

    def test_explicit_estop_wins_over_every_other_phrase(self):
        self.assertEqual(VoiceIntent.ESTOP, parse_intent("急停，停止录制").intent)
        self.assertEqual(VoiceIntent.ESTOP, parse_intent("hello lamp emergency stop").intent)

    def test_phrases_use_the_same_whitespace_normalization_as_transcripts(self):
        self.assertEqual(VoiceIntent.WAKE, parse_intent("Hello Lamp").intent)

    def test_every_allow_listed_phrase_is_reachable(self):
        for intent, phrases in INTENT_PHRASES:
            for phrase in phrases:
                with self.subTest(intent=intent, phrase=phrase):
                    self.assertEqual(intent, parse_intent(phrase).intent)
