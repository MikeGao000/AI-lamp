"""Cloud vision clients. Network access is opt-in through AppConfig."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Protocol
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
    ) -> str:
        if not jpeg:
            raise CloudVisionError("camera returned an empty image")
        return self.text


@dataclass
class OpenAIResponsesVisionClient:
    """Minimal OpenAI Responses API client using a base64 JPEG data URL."""

    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    timeout_s: float = 30.0

    def describe_page(
        self,
        jpeg: bytes,
        prompt: str | None = None,
        system_instructions: str | None = None,
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
        }
        request = Request(
            self.base_url.rstrip("/") + "/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
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
