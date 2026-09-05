import unittest

from lamp_core.cloud import CloudVisionError, OpenAIResponsesVisionClient, StaticStoryClient


class CloudTests(unittest.TestCase):
    def test_static_story_rejects_empty_frame(self):
        with self.assertRaises(CloudVisionError):
            StaticStoryClient().describe_page(b"")

    def test_static_story_accepts_an_optional_prompt(self):
        self.assertEqual(StaticStoryClient().describe_page(b"frame", prompt="read it"), "我看到新的一页了，我们继续读。")

    def test_static_story_accepts_optional_system_instructions(self):
        self.assertEqual(
            StaticStoryClient().describe_page(b"frame", system_instructions="be safe"),
            "我看到新的一页了，我们继续读。",
        )

    def test_extracts_responses_output_text(self):
        response = {"output": [{"type": "message", "content": [{"type": "output_text", "text": "Hello"}]}]}
        self.assertEqual(OpenAIResponsesVisionClient._extract_text(response), "Hello")

    def test_rejects_response_without_text(self):
        with self.assertRaises(CloudVisionError):
            OpenAIResponsesVisionClient._extract_text({"output": []})
