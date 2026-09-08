"""Local picture-book page memory for repeat readings on Raspberry Pi."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


class PageMemoryError(RuntimeError):
    """A page could not be fingerprinted or stored safely."""


@dataclass(frozen=True)
class CachedPage:
    page_id: str
    recognition: dict | None
    speech_segments: tuple[str, ...]
    next_context: str | None
    image_path: Path


def fingerprint_jpeg(jpeg: bytes) -> str:
    """Return a 64-bit difference hash that tolerates small camera variations."""

    if not jpeg:
        raise PageMemoryError("cannot fingerprint an empty image")
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise PageMemoryError("OpenCV and NumPy are required for page memory") from error
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise PageMemoryError("page image is not a decodable JPEG")
    resized = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (resized[:, 1:] > resized[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def fingerprint_distance(left: str, right: str) -> int:
    """Return the Hamming distance between two 64-bit hexadecimal hashes."""

    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except ValueError as error:
        raise PageMemoryError("invalid page fingerprint") from error


class PageMemory:
    """Persist page images and approved narration, then learn visual variants."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        root: str | Path,
        *,
        match_distance: int = 7,
        max_visual_samples: int = 5,
        fingerprint: Callable[[bytes], str] = fingerprint_jpeg,
    ) -> None:
        if not 0 <= match_distance <= 64:
            raise ValueError("match_distance must be between 0 and 64")
        if max_visual_samples < 1:
            raise ValueError("max_visual_samples must be positive")
        self.root = Path(root)
        self.pages_dir = self.root / "pages"
        self.audio_dir = self.root / "audio"
        self.match_distance = match_distance
        self.max_visual_samples = max_visual_samples
        self._fingerprint = fingerprint
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    def find(self, jpeg: bytes) -> CachedPage | None:
        observed = self._fingerprint(jpeg)
        best: tuple[int, Path, dict] | None = None
        for metadata_path in self.pages_dir.glob("*/page.json"):
            try:
                record = json.loads(metadata_path.read_text(encoding="utf-8"))
                fingerprints = tuple(str(item) for item in record["fingerprints"])
                distance = min(fingerprint_distance(observed, item) for item in fingerprints)
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, PageMemoryError):
                continue
            if distance <= self.match_distance and (best is None or distance < best[0]):
                best = (distance, metadata_path, record)
        if best is None:
            return None
        distance, metadata_path, record = best
        self._remember_observation(metadata_path, record, observed, jpeg, distance)
        return self._cached_page(metadata_path.parent, record)

    def is_same_page(self, page_id: str, jpeg: bytes) -> bool:
        """Confirm an interruption ended on the same page and learn safe variants."""

        metadata_path = self.pages_dir / page_id / "page.json"
        try:
            record = json.loads(metadata_path.read_text(encoding="utf-8"))
            observed = self._fingerprint(jpeg)
            fingerprints = tuple(str(item) for item in record["fingerprints"])
            distance = min(fingerprint_distance(observed, item) for item in fingerprints)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, PageMemoryError):
            return False
        if distance > self.match_distance:
            return False
        try:
            self._remember_observation(metadata_path, record, observed, jpeg, distance)
        except OSError:
            # A confirmed visual match remains valid even if the SD card cannot
            # persist this optional additional learning sample.
            pass
        return True

    def store(
        self,
        jpeg: bytes,
        *,
        recognition: dict | None,
        speech_segments: tuple[str, ...],
        next_context: str | None,
    ) -> CachedPage:
        cleaned_segments = tuple(segment.strip() for segment in speech_segments if segment.strip())
        if not cleaned_segments:
            raise PageMemoryError("cannot store a page without spoken content")
        observed = self._fingerprint(jpeg)
        page_id = page_id_for_jpeg(jpeg)
        page_dir = self.pages_dir / page_id
        page_dir.mkdir(parents=True, exist_ok=True)
        image_path = page_dir / "reference.jpg"
        image_path.write_bytes(jpeg)
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "schema_version": self.SCHEMA_VERSION,
            "page_id": page_id,
            "fingerprints": [observed],
            "recognition": recognition,
            "speech_segments": list(cleaned_segments),
            "next_context": next_context,
            "created_at": now,
            "last_seen_at": now,
            "hit_count": 0,
        }
        self._write_record(page_dir / "page.json", record)
        return self._cached_page(page_dir, record)

    def _remember_observation(
        self,
        metadata_path: Path,
        record: dict,
        observed: str,
        jpeg: bytes,
        distance: int,
    ) -> None:
        fingerprints = [str(item) for item in record.get("fingerprints", [])]
        # Only learn close variants.  A looser successful match may be useful
        # for playback, but must not gradually pull this page toward a different
        # illustration after several borderline observations.
        learning_distance = max(2, self.match_distance // 2)
        if (
            distance <= learning_distance
            and observed not in fingerprints
            and len(fingerprints) < self.max_visual_samples
        ):
            sample_number = len(fingerprints) + 1
            (metadata_path.parent / f"sample-{sample_number:02}.jpg").write_bytes(jpeg)
            fingerprints.append(observed)
        record["fingerprints"] = fingerprints
        record["hit_count"] = int(record.get("hit_count", 0)) + 1
        record["last_seen_at"] = datetime.now(timezone.utc).isoformat()
        self._write_record(metadata_path, record)

    @staticmethod
    def _cached_page(page_dir: Path, record: dict) -> CachedPage:
        recognition = record.get("recognition")
        page_id = record.get("page_id")
        raw_segments = record.get("speech_segments")
        if not isinstance(page_id, str) or not page_id:
            raise PageMemoryError("cached page lacks a page id")
        if not isinstance(raw_segments, list) or not raw_segments:
            raise PageMemoryError("cached page lacks spoken content")
        return CachedPage(
            page_id=page_id,
            recognition=recognition if isinstance(recognition, dict) else None,
            speech_segments=tuple(str(item) for item in raw_segments),
            next_context=(str(record["next_context"]) if record.get("next_context") else None),
            image_path=page_dir / "reference.jpg",
        )

    @staticmethod
    def _write_record(path: Path, record: dict) -> None:
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)


def page_id_for_jpeg(jpeg: bytes) -> str:
    """Return the stable identity used before a newly accepted page is stored."""

    if not jpeg:
        raise PageMemoryError("cannot identify an empty image")
    return hashlib.sha256(jpeg).hexdigest()[:16]
