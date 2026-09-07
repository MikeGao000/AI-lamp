import unittest

from lamp_core.reading_prompt import PICTURE_BOOK_SYSTEM_INSTRUCTIONS, build_picture_book_prompt
from app_main import _language_for_voice, page_context_for_next_page, story_extension_for_tts


class ReadingPromptTests(unittest.TestCase):
    def test_prompt_requires_safe_structured_response(self):
        prompt = build_picture_book_prompt("Danish", previous_page_context="Previous page: a bear says goodnight.")
        self.assertIn("Return ONLY one valid JSON object", prompt)
        self.assertIn("visible_text", prompt)
        self.assertIn("narration", prompt)
        self.assertIn("teacher_story", prompt)
        self.assertIn("Previous page: a bear says goodnight.", prompt)
        self.assertIn("READING_POSTURE, GENTLE_NOD, STAY_STILL, or NONE", prompt)
        self.assertIn("approximately 150-200 visible characters", prompt)
        self.assertIn("continuity_callback", prompt)
        self.assertIn("put exactly one short, natural callback", prompt)
        self.assertIn("directly relevant", prompt)
        self.assertIn("about 2 years old", build_picture_book_prompt("Danish"))

    def test_system_prompt_keeps_hardware_out_of_model_output(self):
        self.assertIn("Never output joint angles", PICTURE_BOOK_SYSTEM_INSTRUCTIONS)
        self.assertIn("Do not identify people", PICTURE_BOOK_SYSTEM_INSTRUCTIONS)

    def test_danish_voice_selects_danish_explanations(self):
        self.assertEqual(_language_for_voice("da"), "Danish")

    def test_only_accepted_page_details_are_kept_for_next_page(self):
        context = page_context_for_next_page(
            {"visible_text": "Godnat", "image_description": "A bear under a moon."}
        )
        self.assertIn("Godnat", context)
        self.assertIn("A bear under a moon.", context)
        self.assertLessEqual(len(context), 1200)

    def test_relevant_callback_is_spoken_only_with_prior_page_context(self):
        page = {
            "spoken_reading": "Godnat",
            "teacher_story": "Bjørnen ser op på månen.",
            "continuity_callback": "Se, månen er her igen.",
        }
        self.assertEqual(
            story_extension_for_tts(page, "Previous page: a moon."),
            "Se, månen er her igen. Bjørnen ser op på månen.",
        )
        self.assertEqual(story_extension_for_tts(page), "Bjørnen ser op på månen.")
