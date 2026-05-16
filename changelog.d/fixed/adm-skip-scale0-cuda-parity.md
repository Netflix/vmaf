- **CUDA integer_adm `adm_skip_scale0` option** (`integer_adm_cuda.c`): the
  CUDA backend silently dropped the `adm_skip_scale0` option, always
  accumulating and emitting scale-0 results even when the caller explicitly
  requested scale-0 suppression. Added the option and host-side suppression
  logic to match the CPU reference: when set, scale-0 num/den are excluded
  from the overall ADM score and `integer_adm_scale0` is emitted as 0.0
  (GPU still computes scale 0; suppression is host-side only).
