import unittest

from lamp_core.reading_session import (
    DANISH_RESUME_TRANSITION,
    InterruptibleReadingSession,
    ReadingState,
    split_short_sentences,
)


class ManualTrackedSpeaker:
    def __init__(self) -> None:
        self.jobs: list[dict] = []
        self.started_texts: list[str] = []
        self.interrupts = 0

    def speak_tracked(self, text, *, on_started=None, on_complete=None):
        self.jobs.append(
            {"text": text, "on_started": on_started, "on_complete": on_complete}
        )

    def interrupt(self):
        self.interrupts += 1
        self.jobs.clear()

    def start_next(self) -> str:
        job = self.jobs[0]
        self.started_texts.append(job["text"])
        if job["on_started"]:
            job["on_started"]()
        return job["text"]

    def complete_next(self) -> str:
        job = self.jobs.pop(0)
        if job["on_complete"]:
            job["on_complete"]()
        return job["text"]


class ReadingSessionTests(unittest.TestCase):
    def test_splits_danish_chinese_and_english_at_sentence_boundaries(self):
        self.assertEqual(
            ("Hej måne.", "你看见星星了吗？", "Look up!"),
            split_short_sentences("Hej måne. 你看见星星了吗？ Look up!"),
        )

    def test_interrupt_answers_transitions_then_resumes_at_next_sentence(self):
        speaker = ManualTrackedSpeaker()
        answers = []
        session = InterruptibleReadingSession(
            speaker,
            answerer=lambda jpeg, question, context: answers.append(question) or "这是星星。",
            current_frame=lambda: b"same-page-now",
            same_page=lambda page_id, jpeg: page_id == "page-1" and jpeg == b"same-page-now",
        )
        session.begin_page("page-1", b"accepted-page")
        session.speak("Første sætning. Anden sætning. Tredje sætning.")
        session.bind_page("page-1", b"accepted-page", "three sentences")
        self.assertEqual("Første sætning.", speaker.start_next())
        self.assertEqual(0, session.current_sentence_index)

        self.assertTrue(session.notify_child_speech_started())
        self.assertEqual(ReadingState.LISTENING, session.state)
        self.assertEqual("这是星星。", session.answer_question("请用中文回答"))
        self.assertEqual("这是星星。", speaker.start_next())
        speaker.complete_next()
        self.assertEqual(DANISH_RESUME_TRANSITION, speaker.start_next())
        speaker.complete_next()
        self.assertEqual("Anden sætning.", speaker.start_next())
        self.assertNotIn("Første sætning.", speaker.started_texts[1:])
        self.assertEqual(["请用中文回答"], answers)

    def test_changed_page_blocks_transition_and_resume(self):
        speaker = ManualTrackedSpeaker()
        session = InterruptibleReadingSession(
            speaker,
            answerer=lambda jpeg, question, context: "Det er en stjerne.",
            current_frame=lambda: b"different-page",
            same_page=lambda page_id, jpeg: False,
        )
        session.begin_page("page-1", b"accepted-page")
        session.speak("En. To.")
        session.bind_page("page-1", b"accepted-page", None)
        speaker.start_next()
        session.notify_child_speech_started()
        session.answer_question("Hvad er det?")
        speaker.start_next()
        speaker.complete_next()
        self.assertEqual(ReadingState.PAUSED_PAGE_CHANGED, session.state)
        self.assertEqual([], speaker.jobs)

    def test_question_after_last_sentence_answers_without_false_resume(self):
        speaker = ManualTrackedSpeaker()
        session = InterruptibleReadingSession(
            speaker,
            answerer=lambda jpeg, question, context: "Ja.",
            current_frame=lambda: b"same",
            same_page=lambda page_id, jpeg: True,
        )
        session.begin_page("page-1", b"same")
        session.speak("Slut.")
        session.bind_page("page-1", b"same", None)
        speaker.start_next()
        speaker.complete_next()
        self.assertEqual(ReadingState.FINISHED, session.state)
        session.notify_child_speech_started()
        session.answer_question("Er den slut?")
        speaker.start_next()
        speaker.complete_next()
        self.assertEqual(ReadingState.FINISHED, session.state)
        self.assertEqual([], speaker.jobs)


if __name__ == "__main__":
    unittest.main()
