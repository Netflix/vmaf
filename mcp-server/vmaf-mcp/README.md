# vmaf-mcp

MCP (Model Context Protocol) server that exposes the Lusoris VMAF fork's
scoring CLI to LLM tooling via JSON-RPC over stdio.

## Tools

| Tool            | Description                                                  |
| --------------- | ------------------------------------------------------------ |
| `vmaf_score`    | Score a (ref, dis) YUV pair. Returns the full JSON report.   |
| `list_models`   | Enumerate models under `model/` (`.json`, `.pkl`, `.onnx`).  |
| `list_backends` | Report which backends (`cpu`/`cuda`/`sycl`/`hip`) are live.  |
| `run_benchmark` | Run `testdata/bench_all.sh` on a pair.                       |

## Install

```bash
cd mcp-server/vmaf-mcp
pip install -e .
```

Requires a built `libvmaf` binary at `build/tools/vmaf` (override via
`VMAF_BIN=/abs/path/to/vmaf`).

## Run

```bash
# Stdio transport (default for Claude Desktop, Cursor, etc.)
vmaf-mcp
```

## Path allowlisting

For safety, the server only reads files under `testdata/` and
`python/test/resource/`. Extend via colon-separated `VMAF_MCP_ALLOW`:

```bash
VMAF_MCP_ALLOW=/data/my-corpus:/mnt/yuv vmaf-mcp
```

## Claude Desktop config

```json
{
  "mcpServers": {
    "vmaf": {
      "command": "vmaf-mcp",
      "env": {
        "VMAF_BIN": "/home/you/dev/vmaf/build/tools/vmaf",
        "VMAF_MCP_ALLOW": "/data/yuv-corpus"
      }
    }
  }
}
```
