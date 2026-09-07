import json
import unittest

from app_main import ChildQuestionSession, answer_child_question
from lamp_core.config import AppConfig
from lamp_core.question_prompt import build_child_question_prompt, detect_one_turn_reply_language


class QuestionClient:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses
        self.prompt = ""
        self.prompts: list[str] = []

    def describe_page(self, jpeg, prompt=None, system_instructions=None, on_output_text_delta=None):
        self.prompt = prompt or ""
        self.prompts.append(self.prompt)
        return json.dumps(self.responses.pop(0))


class RecordingSpeaker:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def speak(self, text: str) -> None:
        self.messages.append(text)


def config() -> AppConfig:
    return AppConfig(
        False, None, "https://api.openai.com/v1", "gpt-4o-mini", 1.5, 8.0, "da",
        "local", "gpt-4o-mini-tts", "marin", "warm", 0.94, 30.0, "Danish", "Danish",
        True, "none", 700,
    )


class ChildQuestionTests(unittest.TestCase):
    def test_chinese_question_can_request_a_danish_grounded_answer(self):
        client = QuestionClient([
            {"answer": "Det er en stjerne. Den lyser på nattehimlen.", "reply_language": "Danish"}
        ])
        speaker = RecordingSpeaker()
        result = answer_child_question(
            client, speaker, config(), b"jpeg", "这是什么？", pointed_object="star"
        )
        self.assertEqual("Det er en stjerne. Den lyser på nattehimlen.", result.answer)
        self.assertEqual([result.answer], speaker.messages)
        self.assertIn("这是什么？", client.prompt)
        self.assertIn("Reply in exactly: Danish.", client.prompt)
        self.assertIn("Local pointing hint from the vision layer: star.", client.prompt)

    def test_question_prompt_can_request_chinese_reply(self):
        prompt = build_child_question_prompt("What is this?", "Chinese", "star")
        self.assertIn("Reply in exactly: Chinese.", prompt)
        self.assertIn("star", prompt)

    def test_empty_question_is_rejected_before_a_cloud_call(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            answer_child_question(QuestionClient([]), RecordingSpeaker(), config(), b"jpeg", "  ")

    def test_language_request_rephrases_only_that_turn_and_keeps_default_danish(self):
        client = QuestionClient([
            {"answer": "Det er en stjerne.", "reply_language": "Danish"},
            {"answer": "这是星星。", "reply_language": "Chinese"},
            {"answer": "Den lyser på nattehimlen.", "reply_language": "Danish"},
        ])
        speaker = RecordingSpeaker()
        session = ChildQuestionSession(client, speaker, config())
        first = session.ask(b"jpeg", "这是什么？", pointed_object="star")
        second = session.ask(b"jpeg", "你用中文说。")
        third = session.ask(b"jpeg", "它为什么会亮？", pointed_object="star")
        self.assertEqual("Det er en stjerne.", first.answer)
        self.assertEqual("这是星星。", second.answer)
        self.assertEqual("Den lyser på nattehimlen.", third.answer)
        self.assertEqual([first.answer, second.answer, third.answer], speaker.messages)
        self.assertEqual("Danish", config().question_reply_language)
        self.assertIn("Reply in exactly: Chinese.", client.prompts[1])
        self.assertIn("Immediately previous spoken answer: Det er en stjerne.", client.prompts[1])

    def test_detects_a_one_turn_chinese_request(self):
        self.assertEqual("Chinese", detect_one_turn_reply_language("请你用中文说"))
