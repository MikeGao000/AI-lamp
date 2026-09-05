import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lamp_core.config import load_dotenv


class ConfigTests(unittest.TestCase):
    def test_dotenv_loads_values_without_overriding_existing_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("ONE=loaded\nTWO='quoted value'\n", encoding="utf-8")
            with patch.dict(os.environ, {"ONE": "explicit"}, clear=True):
                load_dotenv(env_file)
                self.assertEqual("explicit", os.environ["ONE"])
                self.assertEqual("quoted value", os.environ["TWO"])
