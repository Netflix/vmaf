"""OllamaClient tests — no network. We monkey-patch urlopen."""

from __future__ import annotations

import io
import json
from unittest.mock import patch

import pytest

from vmaf_dev_llm.ollama_client import OllamaClient, OllamaError


class _FakeResponse:
    def __init__(self, payload: bytes, status: int = 200) -> None:
        self._payload = payload
        self.status = status

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *args):  # noqa: ANN001
        return False


def test_generate_returns_response() -> None:
    client = OllamaClient()
    payload = json.dumps({"response": "hello world"}).encode()
    with patch("urllib.request.urlopen", return_value=_FakeResponse(payload)):
        out = client.generate("qwen2.5-coder:7b", "say hi")
    assert out == "hello world"


def test_generate_raises_on_bad_payload() -> None:
    client = OllamaClient()
    payload = json.dumps({"no_response_field": True}).encode()
    with patch("urllib.request.urlopen", return_value=_FakeResponse(payload)):
        with pytest.raises(OllamaError):
            client.generate("qwen", "x")


def test_available_false_on_error() -> None:
    client = OllamaClient()
    with patch("urllib.request.urlopen", side_effect=OSError("refused")):
        assert client.available() is False


def test_available_true_on_200() -> None:
    client = OllamaClient()
    with patch("urllib.request.urlopen", return_value=_FakeResponse(b"[]")):
        assert client.available() is True
