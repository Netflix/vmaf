"""vmaf-dev-llm CLI — review | commitmsg | docgen."""

from __future__ import annotations

import subprocess
import sys
from importlib import resources
from pathlib import Path

import typer
from rich.console import Console

from .config import Config, load_config
from .ollama_client import OllamaClient, OllamaError

app = typer.Typer(add_completion=False, help="Local LLM dev helpers (Ollama-first).")
console = Console()


EXT_TO_PROMPT = {
    ".c":   "review_c.md",
    ".h":   "review_c.md",
    ".cpp": "review_c.md",
    ".hpp": "review_c.md",
    ".cu":  "review_cuda.md",
    ".cuh": "review_cuda.md",
    ".sycl": "review_sycl.md",
}


def _prompt_dir(cfg: Config) -> Path:
    """Return the directory that holds prompt templates."""
    if cfg.prompts_dir is not None:
        return cfg.prompts_dir
    with resources.as_file(resources.files("vmaf_dev_llm").joinpath("prompts")) as p:
        return p


def _load_prompt(cfg: Config, name: str) -> str:
    path = _prompt_dir(cfg) / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text()


def _guess_prompt_for_file(file_path: Path) -> str:
    if file_path.name.endswith(".sycl.cpp") or "sycl" in file_path.parts:
        return "review_sycl.md"
    return EXT_TO_PROMPT.get(file_path.suffix, "review_c.md")


def _run_ollama(cfg: Config, *, prompt: str, system: str | None = None, model: str | None = None) -> str:
    client = OllamaClient(base_url=cfg.ollama.base_url, timeout_seconds=cfg.ollama.timeout_seconds)
    if not client.available():
        raise OllamaError(
            f"Ollama is not reachable at {cfg.ollama.base_url}. "
            "Start it with `ollama serve` and pull a model first."
        )
    return client.generate(model or cfg.ollama.default_model, prompt, system=system)


@app.command()
def review(
    file: Path = typer.Option(..., "--file", "-f", help="Source file to review.", exists=True, readable=True),
    model: str | None = typer.Option(None, "--model", "-m", help="Override config.ollama.default_model."),
) -> None:
    """Run a local LLM code review against FILE and print review comments."""
    cfg = load_config()
    template = _load_prompt(cfg, _guess_prompt_for_file(file))
    source = file.read_text()
    prompt = template.replace("{{FILE_PATH}}", str(file)).replace("{{SOURCE}}", source)
    try:
        out = _run_ollama(cfg, prompt=prompt, model=model)
    except OllamaError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    console.print(out)


@app.command()
def commitmsg(
    model: str | None = typer.Option(None, "--model", "-m"),
) -> None:
    """Draft a Conventional-Commits message from `git diff --staged`."""
    cfg = load_config()
    try:
        diff = subprocess.check_output(["git", "diff", "--staged", "--no-color"], text=True)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]git diff --staged failed: {e}[/red]")
        sys.exit(1)
    if not diff.strip():
        console.print("[yellow]No staged changes — nothing to draft.[/yellow]")
        sys.exit(2)
    template = _load_prompt(cfg, "commit_msg.md")
    prompt = template.replace("{{DIFF}}", diff)
    try:
        out = _run_ollama(cfg, prompt=prompt, model=model)
    except OllamaError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    # Print without rich formatting so the user can pipe or copy cleanly.
    print(out.strip())


@app.command()
def docgen(
    file: Path = typer.Option(..., "--file", "-f", exists=True, readable=True),
    symbol: str = typer.Option(..., "--symbol", "-s", help="Function or type name to document."),
    model: str | None = typer.Option(None, "--model", "-m"),
) -> None:
    """Draft Doxygen @brief/@param blocks for SYMBOL in FILE."""
    cfg = load_config()
    template = _load_prompt(cfg, "doc_section.md")
    source = file.read_text()
    prompt = (
        template
        .replace("{{FILE_PATH}}", str(file))
        .replace("{{SYMBOL}}", symbol)
        .replace("{{SOURCE}}", source)
    )
    try:
        out = _run_ollama(cfg, prompt=prompt, model=model)
    except OllamaError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    print(out.strip())


@app.command()
def check() -> None:
    """Verify local config + Ollama availability; useful for /doctor flows."""
    cfg = load_config()
    console.print(f"backend: [cyan]{cfg.backend}[/cyan]")
    console.print(f"ollama.base_url: {cfg.ollama.base_url}")
    console.print(f"ollama.default_model: {cfg.ollama.default_model}")
    client = OllamaClient(base_url=cfg.ollama.base_url)
    ok = client.available()
    color = "green" if ok else "red"
    console.print(f"ollama reachable: [{color}]{ok}[/{color}]")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    app()
