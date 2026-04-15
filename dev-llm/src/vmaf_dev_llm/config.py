"""Config loader for vmaf-dev-llm."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    default_model: str = "qwen2.5-coder:7b"
    timeout_seconds: int = 120


@dataclass
class Config:
    backend: str = "ollama"
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    prompts_dir: Path | None = None


def config_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "vmaf-dev-llm" / "config.yaml"


def load_config(path: Path | None = None) -> Config:
    """Load config from disk; return defaults if not present."""
    p = path or config_path()
    if not p.exists():
        return Config()
    raw = yaml.safe_load(p.read_text()) or {}
    ollama_raw = raw.get("ollama") or {}
    prompts_dir = raw.get("prompts_dir")
    return Config(
        backend=raw.get("backend", "ollama"),
        ollama=OllamaConfig(
            base_url=ollama_raw.get("base_url", "http://localhost:11434"),
            default_model=ollama_raw.get("default_model", "qwen2.5-coder:7b"),
            timeout_seconds=int(ollama_raw.get("timeout_seconds", 120)),
        ),
        prompts_dir=Path(prompts_dir).expanduser() if prompts_dir else None,
    )
