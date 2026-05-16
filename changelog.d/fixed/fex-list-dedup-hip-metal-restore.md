fix(fex): restore 61 deduplicated entries + integer_ms_ssim_hip + integer_vif_metal clobbered by PR #1088

PR #1088 (test_version squash) merged from a stale base that predated:
- PR #1085 (remove 61 duplicate SYCL/Vulkan pointer entries from feature_extractor_list[])
- PR #1111 (restore integer_ms_ssim_hip and integer_vif_metal wire-ins dropped by PR #1067)

Net effect: 6 SYCL + 55 Vulkan duplicate extractor registrations were re-introduced,
and the integer_ms_ssim_hip (ADR-0285) and integer_vif_metal (ADR-0436) entries were
silently dropped, making those extractors unreachable by name lookup.
