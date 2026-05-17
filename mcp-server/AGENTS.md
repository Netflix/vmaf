# AGENTS.md — mcp-server/

Orientation for agents working on the MCP (Model Context Protocol) server.
Parent: [../AGENTS.md](../AGENTS.md).

## Scope

A Python JSON-RPC server that exposes libvmaf capabilities as MCP tools
for editor/agent consumers.

```
mcp-server/
  vmaf-mcp/
    pyproject.toml
    src/                    # tool implementations + JSON-RPC glue
    tests/
```

## Exposed tools

Locked in [ADR-0009](../docs/adr/0009-mcp-server-tool-surface.md):

- `vmaf_score` — score a ref/dist pair, returning per-frame + aggregate
- `list_models` — enumerate registered VMAF models (`model/`) + tiny models (`model/tiny/`)
- `list_backends` — SIMD caps + GPU devices present on the host
- `run_benchmark` — run `vmaf_bench` on a fixture set
- `eval_model_on_split` — evaluate a tiny-AI ONNX regressor on a parquet split
- `compare_models` — rank ONNX regressors on the same split
- `describe_worst_frames` — local VLM describes the N frames with lowest VMAF score

## Ground rules

- **Parent rules** apply (see [../AGENTS.md](../AGENTS.md)).
- **Never shell out to `vmaf` with user-controlled args** — the MCP server
  is the trusted front-end; tool arguments are untrusted. Use the Python
  bindings in [../python/vmaf/](../python/vmaf/) or in-process libvmaf via
  ctypes / cffi. If shelling out is unavoidable, pass args as a list and
  validate against an explicit schema.
- **No paths escape the caller's workspace** — any filesystem arg is
  resolved via `realpath` and rejected if it escapes a configured root.
- **Tiny-AI surface rule applies**: MCP tools that touch the tiny-AI path
  (e.g. `describe_worst_frames`) ship docs under `docs/ai/` in the same PR.
  See [ADR-0042](../docs/adr/0042-tinyai-docs-required-per-pr.md).

## Governing ADRs

- [ADR-0005](../docs/adr/0005-framework-adaptation-full-scope.md) — framework scope includes MCP.
- [ADR-0009](../docs/adr/0009-mcp-server-tool-surface.md) — the four initial tools.
- [ADR-0036](../docs/adr/0036-tinyai-wave1-scope-expansion.md) — `describe_worst_frames` on the Wave 1 list.
- [ADR-0042](../docs/adr/0042-tinyai-docs-required-per-pr.md) — doc rule.
