"""Interruptible, sentence-aware picture-book playback state."""

from __future__ import annotations

import re
import threading
from enum import Enum
from typing import Callable, Protocol


DANISH_RESUME_TRANSITION = "Godt spørgsmål. Nu læser vi videre."


class TrackedSpeech(Protocol):
    def speak_tracked(
        self,
        text: str,
        *,
        on_started: Callable[[], None] | None = None,
        on_complete: Callable[[], None] | None = None,
    ) -> None: ...

    def interrupt(self) -> None: ...


class ReadingState(str, Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    READING = "reading"
    LISTENING = "listening"
    ANSWERING = "answering"
    FINISHED = "finished"
    PAUSED_PAGE_CHANGED = "paused_page_changed"


def split_short_sentences(text: str, max_chars: int = 150) -> tuple[str, ...]:
    """Split multilingual narration at natural sentence boundaries."""

    cleaned = " ".join(text.split())
    if not cleaned:
        return ()
    sentences = [
        item.strip()
        for item in re.findall(r'.+?(?:[.!?。！？…]+["”’\']?|$)', cleaned)
        if item.strip()
    ]
    result: list[str] = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            result.append(sentence)
            continue
        # The model normally produces short sentences.  This guarded fallback
        # breaks an unusually long one at a phrase boundary for toddler pacing.
        pieces = re.split(r"(?<=[,，;；:：])\s*", sentence)
        current = ""
        for piece in pieces:
            if current and len(current) + 1 + len(piece) > max_chars:
                result.append(current.strip())
                current = piece
            else:
                current = f"{current} {piece}".strip()
        if current:
            result.append(current.strip())
    return tuple(result)


class InterruptibleReadingSession:
    """Play one sentence at a time and resume after a child's question."""

    def __init__(
        self,
        speaker: TrackedSpeech,
        *,
        answerer: Callable[[bytes, str, str | None], str],
        current_frame: Callable[[], bytes],
        same_page: Callable[[str, bytes], bool],
        transition: str = DANISH_RESUME_TRANSITION,
    ) -> None:
        self._speaker = speaker
        self._answerer = answerer
        self._current_frame = current_frame
        self._same_page = same_page
        self._transition = transition
        self._lock = threading.RLock()
        self._generation = 0
        self._sentences: list[str] = []
        self._queued_index: int | None = None
        self._current_index: int | None = None
        self._next_index = 0
        self._resume_index = 0
        self._page_id: str | None = None
        self._page_jpeg = b""
        self._page_context: str | None = None
        self.state = ReadingState.IDLE

    @property
    def current_sentence_index(self) -> int | None:
        with self._lock:
            return self._current_index

    @property
    def sentences(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._sentences)

    def begin_page(self, page_id: str, jpeg: bytes) -> None:
        """Discard old-page playback before cloud/cached content is appended."""

        self._speaker.interrupt()
        with self._lock:
            self._generation += 1
            self._sentences.clear()
            self._queued_index = None
            self._current_index = None
            self._next_index = 0
            self._resume_index = 0
            self._page_id = page_id
            self._page_jpeg = jpeg
            self._page_context = None
            self.state = ReadingState.PREPARING

    def bind_page(self, page_id: str, jpeg: bytes, page_context: str | None) -> None:
        """Attach the accepted cache identity and enable child interruption."""

        with self._lock:
            self._page_id = page_id
            self._page_jpeg = jpeg
            self._page_context = page_context
            if self.state is ReadingState.PREPARING:
                self.state = (
                    ReadingState.READING
                    if self._current_index is not None or self._queued_index is not None
                    else ReadingState.FINISHED
                )

    def speak(self, text: str) -> None:
        """SpeechSink entry used by the production reading coordinator."""

        new_sentences = split_short_sentences(text)
        if not new_sentences:
            return
        with self._lock:
            self._sentences.extend(new_sentences)
            should_start = (
                self.state in (ReadingState.PREPARING, ReadingState.READING)
                and self._queued_index is None
                and self._current_index is None
                and self._next_index < len(self._sentences)
            )
        if should_start:
            self._queue_next_sentence()

    def notify_child_speech_started(self) -> bool:
        """Stop audible narration immediately and remember the resume sentence."""

        with self._lock:
            if self.state not in (ReadingState.READING, ReadingState.FINISHED):
                return False
            if self._current_index is not None:
                self._resume_index = self._current_index + 1
            elif self._queued_index is not None:
                self._resume_index = self._queued_index
            else:
                self._resume_index = self._next_index
            self.state = ReadingState.LISTENING
            self._queued_index = None
            self._current_index = None
        self._speaker.interrupt()
        return True

    def answer_question(self, question: str) -> str | None:
        """Generate the answer, speak it, then conditionally resume this page."""

        cleaned = question.strip()
        if not cleaned:
            return None
        with self._lock:
            if self.state is not ReadingState.LISTENING:
                return None
            generation = self._generation
            # Prefer the latest frame so a hand/finger that appeared with the
            # question can ground words such as "this". Fall back to the clean
            # accepted page if the camera has no newer frame.
            jpeg = self._current_frame() or self._page_jpeg
            context = self._page_context
            self.state = ReadingState.ANSWERING
        answer = self._answerer(jpeg, cleaned, context).strip()
        if not answer:
            raise ValueError("question answer must not be empty")
        with self._lock:
            if generation != self._generation:
                return None
        self._speaker.speak_tracked(
            answer,
            on_complete=lambda: self._after_answer(generation),
        )
        return answer

    def page_moved(self) -> None:
        self._speaker.interrupt()
        with self._lock:
            self._generation += 1
            self._page_id = None
            self._page_jpeg = b""
            self._page_context = None
            self._sentences.clear()
            self._queued_index = None
            self._current_index = None
            self._next_index = 0
            self.state = ReadingState.IDLE

    def resume_after_question_failure(self) -> None:
        """Continue safely when transcription or cloud Q&A fails."""

        with self._lock:
            if self.state not in (ReadingState.LISTENING, ReadingState.ANSWERING):
                return
            self.state = ReadingState.ANSWERING
            generation = self._generation
        self._after_answer(generation)

    def accepts_questions(self) -> bool:
        with self._lock:
            return self.state in (ReadingState.READING, ReadingState.FINISHED)

    def _queue_next_sentence(self) -> None:
        with self._lock:
            if self.state not in (ReadingState.PREPARING, ReadingState.READING):
                return
            if self._queued_index is not None or self._current_index is not None:
                return
            if self._next_index >= len(self._sentences):
                if self.state is ReadingState.READING:
                    self.state = ReadingState.FINISHED
                return
            index = self._next_index
            generation = self._generation
            self._queued_index = index
        self._speaker.speak_tracked(
            self._sentences[index],
            on_started=lambda: self._sentence_started(generation, index),
            on_complete=lambda: self._sentence_completed(generation, index),
        )

    def _sentence_started(self, generation: int, index: int) -> None:
        with self._lock:
            if generation != self._generation:
                return
            self._queued_index = None
            self._current_index = index

    def _sentence_completed(self, generation: int, index: int) -> None:
        with self._lock:
            if generation != self._generation or self.state not in (
                ReadingState.PREPARING,
                ReadingState.READING,
            ):
                return
            self._current_index = None
            self._next_index = index + 1
        self._queue_next_sentence()

    def _after_answer(self, generation: int) -> None:
        with self._lock:
            if generation != self._generation or self.state is not ReadingState.ANSWERING:
                return
            page_id = self._page_id
            resume_index = self._resume_index
            has_more = resume_index < len(self._sentences)
        current = self._current_frame()
        if not page_id or not current or not self._same_page(page_id, current):
            with self._lock:
                if generation == self._generation:
                    self.state = ReadingState.PAUSED_PAGE_CHANGED
            return
        if not has_more:
            with self._lock:
                if generation == self._generation:
                    self.state = ReadingState.FINISHED
            return
        self._speaker.speak_tracked(
            self._transition,
            on_complete=lambda: self._resume_after_transition(generation, resume_index),
        )

    def _resume_after_transition(self, generation: int, resume_index: int) -> None:
        with self._lock:
            if generation != self._generation or self.state is not ReadingState.ANSWERING:
                return
            self.state = ReadingState.READING
            self._next_index = resume_index
            self._queued_index = None
            self._current_index = None
        self._queue_next_sentence()
