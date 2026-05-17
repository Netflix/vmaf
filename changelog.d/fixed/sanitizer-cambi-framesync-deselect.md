Retired two stale sanitizer deselects from the CI workflow:

- `test_cambi` removed from UBSan `EXCLUDE` — PR #761 (2026-05-11) added a
  `__builtin_cpu_supports("avx2")` runtime gate to
  `test_calculate_c_values_scalar_avx2_parity`, eliminating the SIGILL that
  justified the exclusion. The test now skips the AVX2 leg on non-AVX2 runners
  instead of faulting.
- `test_framesync` removed from TSan `EXCLUDE` — PR #548 (2026-05-09) fixed the
  `SAN-FRAMESYNC-MUTEX-DOMAIN` mutex-domain mismatch; the nightly TSan job was
  green on 2026-05-09 and 2026-05-10 (recorded in `docs/state.md`).

Both tests now run under their respective sanitizer lanes on every push.
