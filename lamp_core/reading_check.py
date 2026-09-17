"""Decide whether a cloud reading may be spoken.

The cloud model reads well but invents: asked to merge four near-identical photos of one
page it returned "De begynte med det lille elefanten" for a page that reads "Nu kan vi
begynde med at lave dejen".

A local verbatim transcript was the first plan and was abandoned on measurement: the
integrated text detector returns *region* boxes, not lines (measured: one box of
63% x 35% of the page at 0.99 confidence, holding both the story text and the
illustration), so a recogniser fed from it produced nothing usable.

The check that remains is self-consistency: read the page twice and compare the
**literal** words. ``visible_text`` is a transcription and should repeat, whereas
``narration`` is generated prose and legitimately differs between calls -- so only the
literal field is compared.
"""

from __future__ import annotations

import re
import unicodedata

_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)


def words(text: str) -> set[str]:
    """Comparable words: case folded, accents kept, punctuation dropped."""

    if not isinstance(text, str) or not text.strip():
        return set()
    folded = unicodedata.normalize("NFC", text).casefold()
    cleaned = _NON_WORD.sub(" ", folded)
    return {word for word in cleaned.split() if len(word) > 1}


def transcript_overlap(first: str, second: str) -> float:
    """Share of the shorter reading's words that the other also contains.

    Scored against the shorter side so that one reading simply catching an extra line
    does not look like disagreement; disagreeing about what *is* on the page does.
    """

    left, right = words(first), words(second)
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def readings_agree(first: str, second: str, *, minimum_overlap: float = 0.6) -> bool:
    """Whether two literal readings of the same page agree."""

    if not words(first) or not words(second):
        return False
    return transcript_overlap(first, second) >= minimum_overlap


def readings_are_trustworthy(
    literal_readings, *, minimum_overlap: float = 0.6
) -> bool:
    """Whether a reading may be spoken, given the literal readings taken for the page.

    Refuses unless two of them agree. One reading is never enough: nothing distinguishes
    it from the invented sentence, which is the whole reason this exists.
    """

    texts = [text for text in (literal_readings or []) if words(text)]
    if len(texts) < 2:
        return False
    return readings_agree(texts[0], texts[1], minimum_overlap=minimum_overlap)
