"""Smoke-test that packaged prompt templates are discoverable and template-
substitutable. Does NOT exercise Ollama."""

from __future__ import annotations

from pathlib import Path

from vmaf_dev_llm.cli import _guess_prompt_for_file, _load_prompt
from vmaf_dev_llm.config import Config


def test_all_prompts_present() -> None:
    cfg = Config()
    for name in (
        "review_c.md",
        "review_cuda.md",
        "review_sycl.md",
        "commit_msg.md",
        "doc_section.md",
    ):
        text = _load_prompt(cfg, name)
        assert text.strip(), f"empty prompt: {name}"


def test_guess_prompt() -> None:
    assert _guess_prompt_for_file(Path("foo.c"))      == "review_c.md"
    assert _guess_prompt_for_file(Path("foo.cpp"))    == "review_c.md"
    assert _guess_prompt_for_file(Path("foo.cu"))     == "review_cuda.md"
    assert _guess_prompt_for_file(Path("foo.cuh"))    == "review_cuda.md"
    assert _guess_prompt_for_file(Path("sycl/foo.cpp")) == "review_sycl.md"
    assert _guess_prompt_for_file(Path("foo.rs"))     == "review_c.md"  # fallback


def test_review_c_has_required_placeholders() -> None:
    cfg = Config()
    text = _load_prompt(cfg, "review_c.md")
    assert "{{FILE_PATH}}" in text
    assert "{{SOURCE}}" in text


def test_commit_msg_has_diff_placeholder() -> None:
    cfg = Config()
    text = _load_prompt(cfg, "commit_msg.md")
    assert "{{DIFF}}" in text


def test_doc_section_has_symbol_placeholder() -> None:
    cfg = Config()
    text = _load_prompt(cfg, "doc_section.md")
    assert "{{SYMBOL}}" in text
    assert "{{FILE_PATH}}" in text
    assert "{{SOURCE}}" in text
