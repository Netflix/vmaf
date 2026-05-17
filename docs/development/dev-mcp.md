# dev-MCP Docker Container

The `dev-MCP` container runs the full VMAF fork inside Docker with all four
GPU backends enabled (CUDA, SYCL, Vulkan, HIP) plus the embedded MCP UDS
server.  It is the standard environment for:

- Live probing of VMAF scores across all backends from a single shell.
- Running the continuous smoke-probe cron (`smoke-probe-cron` service).
- Reproducing build regressions on GPU paths other than the host's primary GPU
  (for example: catching HIP toolchain regressions on an NVIDIA-only host).

The design decision is recorded in [ADR-0435](../adr/0435-local-dev-mcp-container.md).

---

## Prerequisites

### Required

| Component | Version | Notes |
| --- | --- | --- |
| Docker Engine | 26+ | `docker compose` v2 plugin required |
| [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) | latest | Enables `--gpus all` / `runtime: nvidia` for CUDA kernel execution. The container builds and runs *without* it; CUDA feature extractors return `-ENOSYS` at runtime. |

### Optional

| Component | Purpose |
| --- | --- |
| AMD ROCm runtime on host | Run HIP kernels inside the container. Without it, HIP compiles but returns an error at kernel dispatch. |
| Intel oneAPI runtime on host | Run SYCL kernels via Level Zero. Without it, SYCL falls back to the OpenCL CPU device or returns an error. |
| `jq` | Pretty-print probe JSON output on the host. `apt install jq`. |

---

## How to build

Use the provided wrapper from the repository root:

```bash
./dev/scripts/dev-mcp-up.sh
```

Or, to build without starting:

```bash
docker compose --project-directory "$(pwd)" -f dev/docker-compose.yml build
```

> **Important — always pass `--project-directory`.**  Without it, Docker
> Compose v2 sets the project directory to the compose-file's parent (`dev/`),
> causing `context: .` to resolve to `dev/` instead of the repo root.  This
> bypasses the root `.dockerignore` and — on developer machines that hold
> `.corpus/` (up to 781 GB) — sends the entire corpus into the build context,
> accumulating copies in `/var/lib/docker/overlay2/` on every failed build.
> The `dev-mcp-up.sh` wrapper always passes `--project-directory`; the bare
> `docker compose -f` form is unsafe unless run from the repo root with the
> flag explicit.

The first build downloads all GPU SDK layers and compiles libvmaf from source.
Expect 20–40 minutes on a typical workstation; subsequent builds use the
layer cache and take 1–3 minutes when only Python packages change.

---

## How to start

```bash
# CPU + Vulkan/lavapipe only (no GPU passthrough)
./dev/scripts/dev-mcp-up.sh

# With NVIDIA GPU passthrough
NVIDIA_VISIBLE_DEVICES=all CONTAINER_RUNTIME=nvidia \
    ./dev/scripts/dev-mcp-up.sh
```

The `dev-mcp-up.sh` wrapper builds (if needed) then starts:

1. `vmaf-dev-mcp` — primary container; exposes MCP UDS at `/sockets/vmaf-mcp.sock`.
2. `vmaf-smoke-probe-cron` — waits for the primary to be healthy, then probes
   every 15 minutes.

Both services write probe files to `.workingdir/dev-mcp-probes/` on the host.

---

## How to attach

```bash
# Interactive bash shell inside the running dev-mcp container
./dev/scripts/dev-mcp-shell.sh

# Run a specific command
./dev/scripts/dev-mcp-shell.sh vmaf-dev-mcp vmaf --version
./dev/scripts/dev-mcp-shell.sh vmaf-dev-mcp vmaf --list-features
```

Inside the container the full environment is initialised:

- `vmaf` CLI — `/usr/local/bin/vmaf`
- `vmaf-mcp-server` — `/opt/vmaf-venv/bin/vmaf-mcp-server`
- GPU SDKs — `nvcc`, `icpx`, `hipcc` in `PATH`
- testdata — `/workspace/testdata/` (read-only bind mount from host repo)
- models — `/workspace/model/` (read-only)

---

## How to manually probe

Run a single smoke probe outside the cron cycle:

```bash
./dev/scripts/dev-mcp-probe.sh
```

This executes `smoke-probe-loop.sh --once` inside the running container and
writes `probe-<timestamp>.json` to `.workingdir/dev-mcp-probes/`.  If `jq` is
installed on the host the result is pretty-printed to stdout.

---

## How to stop

```bash
# Stop, keep volumes (probe history preserved)
./dev/scripts/dev-mcp-down.sh

# Stop and remove volumes (clears socket volume; probe bind-mount preserved)
./dev/scripts/dev-mcp-down.sh --volumes
```

---

## How to interpret probe outputs

Each probe file follows this schema:

```json
{
  "ts": "2026-05-15T14:30:00Z",
  "host_id": "myhostname:abc123def456",
  "backend_results": {
    "cpu":    { "score": 76.45, "duration_ms": 3200, "error": null },
    "cuda":   { "score": 76.44, "duration_ms":  820, "error": null },
    "sycl":   { "score": null,  "duration_ms":    0, "error": "ENOSYS: no SYCL device" },
    "vulkan": { "score": 76.45, "duration_ms": 1100, "error": null }
  },
  "mcp_results": {
    "list_features": { "feature_count": 14, "duration_ms": 45, "error": null },
    "compute_vmaf":  { "score": 76.45, "duration_ms": 3250, "error": null }
  }
}
```

| Field | Meaning |
| --- | --- |
| `score` | Aggregate VMAF score for the 48-frame 576×324 golden pair. `null` = backend failed. |
| `duration_ms` | Wall-clock time for the full scoring run. |
| `error` | Error message string, or `null` for success. |
| `feature_count` | Number of features returned by the MCP `list_features` tool. |

### Expected values

- CPU score: ~76.45 (matches the Netflix golden pair; exact value varies by model version).
- CUDA / Vulkan scores: within ±0.01 of CPU (numeric parity is not bit-exact —
  see [ADR-0214](../adr/0214-gpu-parity-ci-gate.md)).
- SYCL: `ENOSYS` on hosts without Intel GPU or oneAPI runtime; normal.
- HIP: error on NVIDIA-only hosts; normal.

### Common error patterns

| Error | Cause | Action |
| --- | --- | --- |
| `ENOSYS: no CUDA device` | No NVIDIA GPU or Container Toolkit not installed | Install Container Toolkit and set `NVIDIA_VISIBLE_DEVICES=all` |
| `ENOSYS: no SYCL device` | No Intel GPU / oneAPI runtime | Expected on non-Intel hosts; not a regression |
| `mcp stdio returned empty response` | `vmaf-mcp-server` not in PATH or build failed | Rebuild container; check `docker compose logs dev-mcp` |
| Score drift >0.1 from baseline | Code regression or model change | Run `/validate-scores` skill; check recent commits |

---

## Known limitations

| Limitation | Details |
| --- | --- |
| HIP kernels cannot run on NVIDIA-only hosts | The HIP toolchain in the container compiles and embeds HSACO fat binaries, but the AMD ROCm runtime is not available. Feature extractors return an error at kernel dispatch. The container is still valuable for catching compile-time regressions in HIP paths. |
| Metal is disabled on Linux | `libvmaf` is built with `-Denable_metal=auto`, which resolves to disabled on Linux. Metal kernels require macOS + Apple Silicon. |
| SYCL requires Intel GPU or software emulation | Without the oneAPI Level Zero runtime, SYCL falls back to the OpenCL CPU device (if available) or returns `-ENOSYS`. Performance is significantly lower than on a dedicated Intel GPU. |
| Vulkan lavapipe is CPU-backed | The lavapipe software Vulkan ICD enabled in the container allows Vulkan correctness testing without a physical GPU, but throughput is 3–5× slower than real hardware. |
| First build takes 20–40 minutes | All four GPU SDK layers are fetched during `docker compose build`. Subsequent builds are fast (layer cache). |
