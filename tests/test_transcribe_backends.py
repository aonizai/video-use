import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).parents[1] / "helpers" / "transcribe.py"
SPEC = importlib.util.spec_from_file_location("video_use_transcribe", MODULE_PATH)
assert SPEC and SPEC.loader
transcribe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transcribe)


class BackendResolutionTests(unittest.TestCase):
    def test_explicit_backends_are_preserved(self):
        for backend in ("local", "elevenlabs", "none"):
            with self.subTest(backend=backend):
                self.assertEqual(transcribe.resolve_backend(backend), backend)

    def test_auto_prefers_local(self):
        with patch.object(transcribe, "local_backend_available", return_value=True):
            with patch.object(transcribe, "load_api_key") as load_key:
                self.assertEqual(transcribe.resolve_backend("auto"), "local")
                load_key.assert_not_called()

    def test_auto_falls_back_to_elevenlabs(self):
        with patch.object(transcribe, "local_backend_available", return_value=False):
            with patch.object(transcribe, "load_api_key", return_value="key"):
                self.assertEqual(transcribe.resolve_backend("auto"), "elevenlabs")

    def test_auto_fails_cleanly_when_nothing_is_available(self):
        with patch.object(transcribe, "local_backend_available", return_value=False):
            with patch.object(transcribe, "load_api_key", return_value=None):
                with self.assertRaises(RuntimeError):
                    transcribe.resolve_backend("auto")


class FasterWhisperConversionTests(unittest.TestCase):
    def setUp(self):
        transcribe._LOCAL_MODEL_CACHE.clear()

    def test_local_payload_is_pack_transcripts_compatible(self):
        class FakeModel:
            init_calls = 0

            def __init__(self, *args, **kwargs):
                FakeModel.init_calls += 1

            def transcribe(self, *args, **kwargs):
                words = [
                    types.SimpleNamespace(
                        word="สวัสดี",
                        start=0.10,
                        end=0.50,
                        probability=0.99,
                    ),
                    types.SimpleNamespace(
                        word="ครับ",
                        start=0.55,
                        end=0.80,
                        probability=0.98,
                    ),
                ]
                segment = types.SimpleNamespace(text="สวัสดีครับ", words=words)
                info = types.SimpleNamespace(
                    language="th",
                    language_probability=0.97,
                )
                return [segment], info

        fake_module = types.ModuleType("faster_whisper")
        fake_module.WhisperModel = FakeModel

        with patch.dict(sys.modules, {"faster_whisper": fake_module}):
            payload = transcribe.call_faster_whisper(
                Path("fake.wav"),
                language="th",
                model_name="large-v3",
            )
            # A second call should reuse the cached model instance.
            payload2 = transcribe.call_faster_whisper(
                Path("fake2.wav"),
                language="th",
                model_name="large-v3",
            )

        self.assertEqual(FakeModel.init_calls, 1)
        self.assertEqual(payload["transcription_backend"], "faster-whisper")
        self.assertEqual(payload["language_code"], "th")
        self.assertEqual(payload["model_id"], "large-v3")
        self.assertEqual(len(payload["words"]), 2)
        self.assertEqual(payload["words"][0]["type"], "word")
        self.assertIsNone(payload["words"][0]["speaker_id"])
        self.assertEqual(payload2["words"][1]["text"], "ครับ")


if __name__ == "__main__":
    unittest.main()
