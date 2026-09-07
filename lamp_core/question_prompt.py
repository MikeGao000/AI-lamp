"""Prompt contract for a child's question about the current picture-book page."""

from __future__ import annotations


CHILD_QUESTION_SYSTEM_INSTRUCTIONS = """You are Pipi, a gentle picture-book companion for a child aged 3-7.
Answer only what is grounded in the current image, the accepted page context, and the
local pointing hint. The child may ask in one language but the answer language is
always explicitly supplied. Do not identify real people, invent hidden events, or
claim certainty when the object is unclear. Never output hardware commands."""


def detect_one_turn_reply_language(question: str) -> str | None:
    """Recognize a direct request to repeat only this answer in another language.

    This is deliberately a narrow local intent check.  It does not change the
    session preference; the caller uses the result only for the current turn.
    More languages can later be added through the microphone intent adapter.
    """

    normalized = " ".join(question.lower().split())
    language_requests = {
        "Chinese": ("用中文说", "用中文讲", "中文说", "中文回答", "说中文", "讲中文"),
        "Danish": ("用丹麦语说", "用丹麦语讲", "丹麦语回答", "på dansk", "dansk tak"),
        "English": ("用英语说", "用英文说", "英语回答", "in english", "english please"),
    }
    for language, phrases in language_requests.items():
        if any(phrase in normalized for phrase in phrases):
            return language
    return None


def build_child_question_prompt(
    question: str,
    reply_language: str,
    pointed_object: str | None = None,
    accepted_page_context: str | None = None,
    previous_answer: str | None = None,
    is_language_rephrase: bool = False,
) -> str:
    """Build a bounded, multilingual question prompt for the current page only."""

    pointing_hint = pointed_object.strip() if pointed_object and pointed_object.strip() else "none"
    context = accepted_page_context.strip() if accepted_page_context and accepted_page_context.strip() else "none"
    previous = previous_answer.strip() if previous_answer and previous_answer.strip() else "none"
    rephrase_rule = (
        "This is a request to restate the immediately previous answer in the requested "
        "language. Preserve its grounded meaning, answer naturally, and do not change "
        "the session's default language."
        if is_language_rephrase
        else "This is a new question about the current page."
    )
    return f"""A child is looking at the current picture-book page and asks this question:
{question.strip()}

The child may have spoken any language. Reply in exactly: {reply_language}.
Local pointing hint from the vision layer: {pointing_hint}.
Previously accepted page context: {context}.
Immediately previous spoken answer: {previous}.
Turn type: {rephrase_rule}

Return ONLY one valid JSON object in exactly this schema:
{{
  "answer": "one to three short, warm sentences suitable for speaking aloud",
  "reply_language": "{reply_language}",
  "grounded_reference": "the visible object or detail used for the answer, or null",
  "confidence": "high, medium, or low",
  "uncertainty": "short clarification when needed, or null"
}}

Rules:
1. Answer the child's question directly and briefly before adding any optional gentle detail.
2. Treat the pointing hint as a helpful local observation, not proof. Check it against the image.
3. If the object is unclear, say so simply and invite the child to point again; never guess.
4. Do not resume or repeat the whole page story. Do not create new characters, plot, or facts.
5. Keep the answer naturally speakable by the existing picture-book voice."""
