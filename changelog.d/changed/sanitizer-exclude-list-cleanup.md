Removed stale sanitizer deselect patterns from CI workflow. All three sanitizer-excluded tests now pass cleanly:

- **ASan**: `test_model` — fixed by PR #548 (SAN-MODEL-MALLOC-OOB bounds validation)
- **UBSan**: `test_model` and `test_cambi` — fixed by PR #548 (model) and PR #761 (cambi AVX2 runtime gate)
- **TSan**: `test_model` and `test_framesync` — fixed by PR #548 (SAN-FRAMESYNC-MUTEX-DOMAIN lock ordering + SAN-MODEL-MALLOC-OOB)

Verified locally via sanitizer builds (ASan, UBSan, TSan) confirming all tests pass without excluding any test cases.
