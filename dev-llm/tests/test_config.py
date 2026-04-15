"""Config loader tests."""

from pathlib import Path

from vmaf_dev_llm.config import Config, load_config


def test_defaults_when_missing(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "nonexistent.yaml")
    assert isinstance(cfg, Config)
    assert cfg.backend == "ollama"
    assert cfg.ollama.base_url.startswith("http://localhost")
    assert cfg.ollama.default_model.startswith("qwen")
    assert cfg.prompts_dir is None


def test_loads_yaml(tmp_path: Path) -> None:
    p = tmp_path / "config.yaml"
    p.write_text(
        "backend: ollama\n"
        "ollama:\n"
        "  base_url: http://foo:9999\n"
        "  default_model: llama3:70b\n"
        "  timeout_seconds: 30\n"
        "prompts_dir: /tmp/custom\n"
    )
    cfg = load_config(p)
    assert cfg.backend == "ollama"
    assert cfg.ollama.base_url == "http://foo:9999"
    assert cfg.ollama.default_model == "llama3:70b"
    assert cfg.ollama.timeout_seconds == 30
    assert cfg.prompts_dir == Path("/tmp/custom")
