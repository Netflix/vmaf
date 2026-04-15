# Getting support

This is a **community-maintained fork** of Netflix's VMAF. There is no paid
support tier. If you need help, start here:

## Is your question about *upstream* VMAF?

If your question would apply to vanilla Netflix/vmaf (metric correctness,
general VMAF theory, training methodology, published models), the best
place is [upstream Netflix/vmaf](https://github.com/Netflix/vmaf/issues).
We track upstream but don't duplicate its discussion threads.

## Is it about this fork specifically?

- **GPU backends (SYCL / CUDA / HIP)** — [open a bug report](https://github.com/lusoris/vmaf/issues/new/choose)
- **SIMD paths (AVX2 / AVX-512 / NEON)** — same
- **`--precision` flag / IEEE-754 output** — same
- **Tiny-AI surface / ONNX Runtime integration** — same
- **MCP server (`vmaf-mcp`)** — same
- **Build / package / distro issues** — same

## Security issues

**Do not** open a public issue. See [SECURITY.md](SECURITY.md) — use the
private advisory flow or email lusoris@pm.me.

## How to ask a good question

Include:

- Output of `build/tools/vmaf --version` (or `vmaf --version` if installed).
- Host OS + GPU + driver versions if it's a GPU-path question.
- Minimal reproducer: command line + smallest YUV pair (or a public corpus
  ref like a Netflix test fixture) that shows the behavior.
- What you expected vs what you got.
- What you've already tried (different backend, precision, model, etc.).

Questions that include all five usually get triaged within a few days.

## Chat / real-time

None currently. If there's demand we'll open a Discord. Watch
[the repo](https://github.com/lusoris/vmaf) for announcements.

## Sponsoring

If the fork saves you time, [Ko-fi](https://ko-fi.com/lusoris) keeps the
lights on.
