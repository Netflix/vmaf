"""`vmaf-train` command-line entry point.

Subcommands:
  extract-features   run libvmaf on a dataset and dump features → parquet
  fit                train a model from a YAML config
  export             export a trained checkpoint to ONNX (with roundtrip)
  eval               evaluate an ONNX model on a feature parquet (PLCC/SROCC/RMSE)
  register           write the sidecar metadata JSON for a shipped model
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


@app.command("extract-features")
def extract_features_cmd(
    dataset: str = typer.Option(..., help="Dataset name (nflx, konvid-1k, ...)"),
    output: Path = typer.Option(..., help="Output parquet path"),
    vmaf_binary: Path = typer.Option(Path("vmaf"), help="Path to the vmaf CLI binary"),
) -> None:
    """Dump per-frame libvmaf features → parquet for C1 training."""
    from .data.datasets import load_manifest
    from .data.feature_dump import DEFAULT_FEATURES, Entry, dump_features

    manifest = load_manifest(dataset)
    if not manifest:
        console.print(f"[yellow]Manifest for '{dataset}' is empty — nothing to extract.[/yellow]")
        raise typer.Exit(code=2)

    entries = [
        Entry(
            key=m.key,
            ref=Path(m.path),
            dis=Path(m.path),
            width=0,
            height=0,
            mos=m.mos,
        )
        for m in manifest
    ]
    out = dump_features(entries, output, vmaf_binary=vmaf_binary, features=DEFAULT_FEATURES)
    console.print(f"[green]Wrote features to {out}[/green]")


@app.command("fit")
def fit_cmd(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path"),
    cache: Optional[Path] = typer.Option(None, help="Override features cache (.parquet / .npz)"),
    output: Optional[Path] = typer.Option(None, help="Override output run directory"),
    epochs: Optional[int] = typer.Option(None),
    seed: Optional[int] = typer.Option(None),
) -> None:
    """Train a model from a YAML config."""
    from .train import load_config, train

    overrides: dict[str, object] = {}
    if cache is not None:
        overrides["cache"] = str(cache)
    if output is not None:
        overrides["output"] = str(output)
    if epochs is not None:
        overrides["epochs"] = epochs
    if seed is not None:
        overrides["seed"] = seed

    cfg = load_config(config, overrides=overrides)
    ckpt = train(cfg)
    console.print(f"[green]Training done. Last checkpoint: {ckpt}[/green]")


@app.command("export")
def export_cmd(
    checkpoint: Path = typer.Option(..., exists=True, help="Lightning checkpoint"),
    output: Path = typer.Option(..., help="Output .onnx path"),
    model: str = typer.Option("fr_regressor", help="Model family: fr_regressor|nr_metric|learned_filter"),
    opset: int = typer.Option(17),
    atol: float = typer.Option(1e-5, help="Roundtrip tolerance (torch vs onnxruntime)"),
) -> None:
    """Export a trained checkpoint to ONNX with a roundtrip validation step."""
    from .models import export_to_onnx
    from .train import MODEL_REGISTRY

    if model not in MODEL_REGISTRY:
        console.print(f"[red]unknown model family: {model}[/red]")
        raise typer.Exit(code=2)
    model_cls = MODEL_REGISTRY[model]
    loaded = model_cls.load_from_checkpoint(str(checkpoint)).eval()
    input_name = "features" if model == "fr_regressor" else "input"
    output_name = "score" if model in ("fr_regressor", "nr_metric") else "output"
    export_to_onnx(
        loaded,
        output,
        input_name=input_name,
        output_name=output_name,
        opset=opset,
        atol=atol,
    )
    console.print(f"[green]Wrote {output} (opset={opset}, roundtrip atol={atol:g})[/green]")


@app.command("eval")
def eval_cmd(
    model: Path = typer.Option(..., exists=True, help="ONNX model"),
    features: Path = typer.Option(..., exists=True, help="Feature parquet"),
    split: str = typer.Option("test", help="train|val|test"),
    input_name: str = typer.Option("features"),
) -> None:
    """Evaluate an ONNX model on a held-out split."""
    import pandas as pd

    from .data.splits import split_keys
    from .datamodule import FEATURE_COLUMNS
    from .eval import evaluate_onnx

    df = pd.read_parquet(features)
    if "key" in df.columns:
        keys = sorted(df["key"].astype(str).unique())
        splits = split_keys(keys)
        chosen = getattr(splits, split)
        df = df[df["key"].isin(set(chosen))]
    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    x = df[cols].to_numpy()
    y = df["mos"].to_numpy()
    report = evaluate_onnx(model, x, y, input_name=input_name)
    console.print(f"[cyan]{report.pretty()}[/cyan]")


@app.command("manifest-scan")
def manifest_scan_cmd(
    dataset: str = typer.Option(..., help="Dataset name (nflx, konvid-1k, ...)"),
    root: Path = typer.Option(..., exists=True, file_okay=False, help="Local dataset root"),
    mos_csv: Optional[Path] = typer.Option(
        None, "--mos-csv", exists=True, dir_okay=False,
        help="Optional CSV with columns key,mos",
    ),
) -> None:
    """Populate a dataset manifest from a local cache.

    Scans @p root for YUV/Y4M/MP4/MKV/WebM files, pins each by SHA-256, and
    joins an optional MOS CSV (columns: key,mos). Overwrites the in-tree
    manifest file for @p dataset.
    """
    from .data.manifest_scan import scan, write_manifest

    entries = scan(dataset, root, mos_csv)
    if not entries:
        console.print(f"[yellow]No video files found under {root}[/yellow]")
        raise typer.Exit(code=2)
    dst = write_manifest(dataset, entries)
    with_mos = sum(1 for e in entries if e.mos is not None)
    console.print(
        f"[green]Wrote {dst} with {len(entries)} entries "
        f"({with_mos} with MOS)[/green]"
    )


@app.command("register")
def register_cmd(
    model: Path = typer.Option(..., exists=True, help="ONNX model to register"),
    kind: str = typer.Option(..., help="fr|nr|filter"),
    dataset: Optional[str] = typer.Option(None),
    license: Optional[str] = typer.Option(None, "--license"),
    train_commit: Optional[str] = typer.Option(None),
    train_config: Optional[Path] = typer.Option(None, exists=True),
    manifest: Optional[Path] = typer.Option(None, exists=True),
) -> None:
    """Write the sidecar metadata JSON alongside a shipped ONNX model."""
    from .registry import register

    sidecar = register(
        onnx_path=model,
        kind=kind,
        dataset=dataset,
        license_=license,
        train_commit=train_commit,
        train_config=train_config,
        manifest=manifest,
    )
    console.print(f"[green]Wrote sidecar {sidecar}[/green]")


if __name__ == "__main__":
    app()
