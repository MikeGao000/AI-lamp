"""Cloud vision clients. Network access is opt-in through AppConfig."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Callable, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class CloudVisionError(RuntimeError):
    pass


class StoryClient(Protocol):
    def describe_page(
        self,
        jpeg: bytes,
        prompt: str | None = None,
        system_instructions: str | None = None,
        on_output_text_delta: Callable[[str], None] | None = None,
    ) -> str: ...


@dataclass
class StaticStoryClient:
    """Offline story response for simulations and camera tests."""

    text: str = "我看到新的一页了，我们继续读。"

    def describe_page(
        self,
        jpeg: bytes,
        prompt: str | None = None,
        system_instructions: str | None = None,
        on_output_text_delta: Callable[[str], None] | None = None,
    ) -> str:
        if not jpeg:
            raise CloudVisionError("camera returned an empty image")
        if on_output_text_delta is not None:
            on_output_text_delta(self.text)
        return self.text


@dataclass
class OpenAIResponsesVisionClient:
    """Minimal OpenAI Responses API client using a base64 JPEG data URL."""

    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    timeout_s: float = 30.0
    stream: bool = True
    reasoning_effort: str = "none"
    max_output_tokens: int = 700

    def describe_page(
        self,
        jpeg: bytes,
        prompt: str | None = None,
        system_instructions: str | None = None,
        on_output_text_delta: Callable[[str], None] | None = None,
    ) -> str:
        if not jpeg:
            raise CloudVisionError("camera returned an empty image")
        image_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
        payload = {
            "model": self.model,
            "store": False,
            "instructions": system_instructions
            or (
                "You are Pipi, a gentle reading companion for a child. "
                "Describe the pictured book page accurately and briefly. "
                "Reply in the language requested by the child; do not invent unreadable text."
            ),
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt
                            or "Please tell a short, warm story about this page in Danish.",
                        },
                        {"type": "input_image", "image_url": image_url, "detail": "low"},
                    ],
                }
            ],
            "max_output_tokens": self.max_output_tokens,
            "text": {"format": {"type": "json_object"}},
        }
        # gpt-4o-mini does not need reasoning.  Avoid sending a reasoning
        # option unless the caller explicitly selected one for a model that
        # supports it.
        if self.reasoning_effort and self.reasoning_effort.lower() != "none":
            payload["reasoning"] = {"effort": self.reasoning_effort}
        use_stream = self.stream
        if use_stream:
            payload["stream"] = True
        request = Request(
            self.base_url.rstrip("/") + "/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                if use_stream:
                    return self._read_stream(response, on_output_text_delta)
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raise CloudVisionError(f"cloud API returned HTTP {error.code}") from error
        except TimeoutError as error:
            # A response can time out after the request was accepted upstream.  Do
            # not retry here: that could create two billable model requests.  The
            # caller can safely wait for the next stable camera frame and try once.
            raise CloudVisionError(
                f"cloud API response timed out after {self.timeout_s:g}s; retry the page"
            ) from error
        except URLError as error:
            raise CloudVisionError("cloud API is unavailable") from error
        return self._extract_text(body)

    @staticmethod
    def _read_stream(response, on_output_text_delta: Callable[[str], None] | None) -> str:
        """Collect Responses API SSE output while exposing safe text deltas early."""

        chunks: list[str] = []
        terminal_event: str | None = None
        for raw_line in response:
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data:"):
                continue
            try:
                event = json.loads(line.removeprefix("data:").strip())
            except json.JSONDecodeError as error:
                raise CloudVisionError("cloud API sent malformed stream data") from error
            event_type = event.get("type", "")
            if event_type == "response.output_text.delta":
                delta = event.get("delta", "")
                if isinstance(delta, str) and delta:
                    chunks.append(delta)
                    if on_output_text_delta is not None:
                        on_output_text_delta(delta)
            elif event_type in {"response.completed", "response.incomplete", "response.failed"}:
                terminal_event = event_type

        if terminal_event != "response.completed":
            raise CloudVisionError(f"cloud stream ended with {terminal_event or 'no terminal event'}")
        text = "".join(chunks).strip()
        if not text:
            raise CloudVisionError("cloud API stream contained no text")
        return text

    @staticmethod
    def _extract_text(response: dict) -> str:
        if isinstance(response.get("output_text"), str) and response["output_text"].strip():
            return response["output_text"].strip()
        for item in response.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text", "").strip():
                    return content["text"].strip()
        raise CloudVisionError("cloud API response contained no text")
