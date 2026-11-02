import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lamp_core.config import AppConfig, load_dotenv

ENV_EXAMPLE = Path(__file__).resolve().parent.parent / ".env.example"


class ConfigTests(unittest.TestCase):
    def test_env_example_matches_code_defaults(self) -> None:
        # A fresh machine starts from `cp .env.example .env`; it must behave like no .env at all.
        with patch.dict(os.environ, {}, clear=True):
            defaults = AppConfig.from_environment()
            load_dotenv(ENV_EXAMPLE)
            self.assertEqual(defaults, AppConfig.from_environment())

    def test_env_example_covers_every_config_variable(self) -> None:
        source = (ENV_EXAMPLE.parent / "lamp_core" / "config.py").read_text(encoding="utf-8")
        used = set(re.findall(r'os\.getenv\(\s*"([A-Z0-9_]+)"', source))
        listed = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)[=\s]", ENV_EXAMPLE.read_text(encoding="utf-8"), re.M))
        self.assertEqual(set(), used - listed)

    def test_dotenv_loads_values_without_overriding_existing_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("ONE=loaded\nTWO='quoted value'\n", encoding="utf-8")
            with patch.dict(os.environ, {"ONE": "explicit"}, clear=True):
                load_dotenv(env_file)
                self.assertEqual("explicit", os.environ["ONE"])
                self.assertEqual("quoted value", os.environ["TWO"])
