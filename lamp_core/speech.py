"""Local speech adapters; no cloud speech service is required."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EspeakSpeech:
    voice: str = "da"

    def speak(self, text: str) -> None:
        executable = shutil.which("espeak-ng") or shutil.which("espeak")
        if executable is None:
            raise RuntimeError("install espeak-ng to enable local speech")
        subprocess.run([executable, "-v", self.voice, text], check=True)


@dataclass
class WavFileSpeech:
    """The same local TTS engine as the Pi app, rendered to a file for testing."""

    output_path: Path
    voice: str = "da"

    def speak(self, text: str) -> None:
        executable = shutil.which("espeak-ng") or shutil.which("espeak")
        if executable is None:
            raise RuntimeError("install espeak-ng to render a WAV file")
        subprocess.run([executable, "-v", self.voice, "-w", str(self.output_path), text], check=True)
