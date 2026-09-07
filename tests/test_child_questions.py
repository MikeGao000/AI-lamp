import json
import unittest

from app_main import answer_child_question
from lamp_core.config import AppConfig
from lamp_core.question_prompt import build_child_question_prompt


class QuestionClient:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.prompt = ""

    def describe_page(self, jpeg, prompt=None, system_instructions=None, on_output_text_delta=None):
        self.prompt = prompt or ""
        return json.dumps(self.response)


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
        client = QuestionClient(
            {"answer": "Det er en stjerne. Den lyser på nattehimlen.", "reply_language": "Danish"}
        )
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
            answer_child_question(QuestionClient({}), RecordingSpeaker(), config(), b"jpeg", "  ")
