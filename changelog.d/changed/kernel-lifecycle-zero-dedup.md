HIP and Metal kernel-template init functions now use `VMAF_LIFECYCLE_ZERO` (a
shared `memset`-backed macro in `libvmaf/src/kernel_lifecycle_common.h`) instead
of field-by-field zero assignment. New fields added to any lifecycle or
readback/buffer struct are automatically zero-initialised without requiring
manual updates to both backends' init functions (ADR-0485).
