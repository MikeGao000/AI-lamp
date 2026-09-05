import unittest

from lamp_core.voice import VoiceIntent, parse_intent


class VoiceTests(unittest.TestCase):
    def test_known_chinese_command_is_allow_listed(self):
        self.assertEqual(VoiceIntent.READ, parse_intent("小灯，继续读书").intent)
        self.assertEqual(VoiceIntent.STOP, parse_intent("请停止").intent)

    def test_unknown_voice_command_does_not_trigger_motion(self):
        self.assertEqual(VoiceIntent.NONE, parse_intent("今天天气真好").intent)

    def test_teach_commands_are_distinct(self):
        self.assertEqual(VoiceIntent.TEACH_START, parse_intent("开始示教").intent)
        self.assertEqual(VoiceIntent.TEACH_STOP, parse_intent("保存动作").intent)
