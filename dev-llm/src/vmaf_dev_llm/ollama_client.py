"""Minimal Ollama HTTP client using stdlib urllib (no third-party HTTP dep)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass


class OllamaError(RuntimeError):
    pass


@dataclass
class OllamaClient:
    base_url: str = "http://localhost:11434"
    timeout_seconds: int = 120

    def generate(self, model: str, prompt: str, *, system: str | None = None) -> str:
        """One-shot text completion via /api/generate.

        Ollama returns a JSON object with a `response` key when `stream=False`.
        """
        body: dict[str, object] = {"model": model, "prompt": prompt, "stream": False}
        if system:
            body["system"] = system
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/api/generate",
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as r:
                payload = json.loads(r.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise OllamaError(f"Ollama request failed: {e}") from e
        except json.JSONDecodeError as e:
            raise OllamaError(f"Ollama response was not JSON: {e}") from e
        response = payload.get("response")
        if not isinstance(response, str):
            raise OllamaError(f"Ollama response missing `response` field: {payload!r}")
        return response

    def available(self) -> bool:
        """Probe /api/tags; True if Ollama is reachable, False otherwise."""
        try:
            req = urllib.request.Request(f"{self.base_url.rstrip('/')}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status == 200
        except Exception:
            return False
