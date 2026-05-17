Confirmed `VmafCudaKernelLifecycle`, `VmafCudaKernelReadback`, and their associated
helpers (`vmaf_cuda_kernel_lifecycle_init`, `_close`, `vmaf_cuda_kernel_readback_alloc`,
`_free`, `vmaf_cuda_kernel_submit_pre_launch`, `_post_record`, `vmaf_cuda_kernel_collect_wait`)
are intact in `libvmaf/src/cuda/kernel_template.h` and compile cleanly. The container
build cascade regressions (dev-mcp PRs #1192-#1203) were caused by missing
nv-codec-headers, an incorrect `meson setup` working directory, and SYCL
`std::powf`/duplicate-field compilation errors -- all resolved. No lifecycle-helper
code was ever missing from master.
