### `float_ansnr`: restore `enable_chroma` option clobbered by PR #1067

PR #1067 (bootstrap-name-builder dedup refactor) inadvertently replaced the
`enable_chroma`-aware `float_ansnr.c` that PR #947 had introduced with the
pre-#947 luma-only version. The struct field (`bool enable_chroma`), the
`options[]` table entry, the per-plane name arrays (`ansnr_name[3]`,
`anpsnr_name[3]`), the YUV400P clamp in `init()`, and the plane loop in
`extract()` are fully restored. Luma-only behaviour when `enable_chroma=false`
(the default) is unchanged.
