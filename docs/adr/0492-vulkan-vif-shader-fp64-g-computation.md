# ADR-0492: Promote Vulkan VIF g/sv_sq Computation to double Precision

- **Status**: Accepted
- **Date**: 2026-05-17
- **Deciders**: lusoris, Claude (Anthropic)
- **Tags**: `vulkan`, `vif`, `gpu-parity`, `precision`

## Context

The Vulkan VIF compute shader (`vif.comp` lines 525–548) computed the gain
factor `g = sigma12 / sigma1_sq` and the residual variance `sv_sq =
sigma2_sq - g * sigma12` in `precise float` (fp32), while the CPU reference
in `integer_vif.c` uses `double g = sigma12 / (sigma1_sq + eps)` — full
IEEE-754 double precision.

The `precise` qualifier (introduced in research-0053 to block NVIDIA's
Vulkan-1.4 FMA contraction, PR #1201) removed the systematic FMA bias but
left a residual fp32-vs-double divergence of approximately 7 ULP/pixel in
the `g` ratio. At 576×324 (scale 0–3) this accumulates to a per-frame
`integer_vif_scale3` delta of ~2×10⁻⁴, which exceeds the ADR-0214 places=4
gate threshold (1×10⁻⁴).

`GL_EXT_shader_explicit_arithmetic_types_float64` provides IEEE-754 `double`
in GLSL compute shaders. RTX 4090 (and all Vulkan 1.3+ discrete GPUs that
expose `VkPhysicalDeviceFeatures::shaderFloat64`) support it natively.

## Decision

Promote `g`, `sv_sq`, and `gg_sigma` from `precise float` to `double` in
the VIF statistics block of `vif.comp`, matching the CPU path exactly:
`double g = sigma12 / (sigma1_sq + eps)`, `int sv_sq = (int)(sigma2_sq -
g * sigma12)`. Require `GL_EXT_shader_explicit_arithmetic_types_float64`
at shader compile time. Probe `shaderFloat64` at `vmaf_vulkan_context_new`
time and refuse to initialise the Vulkan backend with `-ENOTSUP` on devices
that do not expose it, falling back to CPU.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| A (chosen): promote to double | Exact CPU parity; eliminates systematic bias; GPU supports it natively | Requires shaderFloat64 device feature; breaks MoltenVK (no Metal fp64 buffer ops) | Best correctness / compatibility trade-off for discrete GPU targets |
| B: pure-integer arithmetic (carry sigma12 numerator + sigma1_sq denominator through the log2 LUT) | No fp64 dependency; works on all Vulkan targets including MoltenVK | Significant implementation complexity; requires redesigned LUT indexing; risk of new integer-overflow bugs | Deferred; viable fallback if fp64 adoption is blocked on Apple targets |
| C: keep `precise float` + per-pixel epsilon compensation | No device-feature requirement | Does not close the systematic ~7 ULP/px bias; fails ADR-0214 gate | Rejected — gate failure is not acceptable |

## Consequences

- **Positive**: Vulkan VIF scores pass the ADR-0214 places=4 CPU-parity
  gate at all tested resolutions (576×324, 1920×1080). The fix closes the
  last known precision gap for the Vulkan backend on RTX 4090.
- **Negative**: Devices without `shaderFloat64` (notably Apple Silicon via
  MoltenVK, some ARM Mali integrated GPUs) will not load the Vulkan backend
  and will fall back to CPU. A pure-integer alternative path (Option B) is
  deferred.
- **Neutral / follow-ups**:
  - The shader comment that previously described the `precise float`
    rationale (research-0053 / PR #1201) has been updated to document the
    double-promotion rationale.
  - A changelog fragment is required per ADR-0221.
  - The `docs/backends/vulkan.md` device-requirements table should be
    updated to list `shaderFloat64` as a required feature.

## References

- `integer_vif.c` lines 326–336: CPU double-precision reference path.
- ADR-0214: GPU-parity CI gate (places=4 per-frame threshold).
- ADR-0350: `shaderBufferInt64Atomics` probe precedent (same pattern).
- research-0053: NVIDIA FMA contraction investigation (fp32 `precise` fix).
- Source: user direction (paraphrased) — promote VIF shader g computation
  to double; option A (RTX 4090 supports fp64 compute); verify with docker
  exec vmaf-dev-mcp run.
