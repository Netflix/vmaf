# GPU per-feature kernel scaffolding templates

Status: introduced 2026-04-29 ([ADR-0246](../adr/0246-gpu-kernel-template.md));
templates only — no kernel migrations yet.

This page documents the **per-backend kernel scaffolding templates** that
sit alongside the CUDA, Vulkan, and HIP backend runtimes:

- [`libvmaf/src/cuda/kernel_template.h`](../../libvmaf/src/cuda/kernel_template.h)
- [`libvmaf/src/vulkan/kernel_template.h`](../../libvmaf/src/vulkan/kernel_template.h)
- [`libvmaf/src/hip/kernel_template.h`](../../libvmaf/src/hip/kernel_template.h)
  (T7-10 / [ADR-0241](../adr/0241-hip-first-consumer-psnr.md);
  field-for-field mirror of the CUDA template, helper bodies return
  `-ENOSYS` until the HIP runtime PR T7-10b lands)

These headers exist to absorb the lifecycle boilerplate that every fork-added
GPU feature kernel currently re-implements by hand. They are **template-only**:
no kernel includes them today. Each future migration is a separate PR with its
own `places=4` cross-backend gate (per
[ADR-0214](../adr/0214-gpu-parity-ci-gate.md)).

If you are writing a brand-new GPU feature kernel, prefer the templates over
copy-paste from a neighbouring kernel — the helpers wrap the steps that
historically caused regressions (forgetting `cuStreamSynchronize` before
`cuStreamDestroy`, leaking a `VkDescriptorPool` on a partial-init failure,
etc.).

## CUDA template

The CUDA template formalises the async-stream + event lifecycle every
fork-added CUDA kernel currently uses. The reference implementation is
[`integer_psnr_cuda.c`](../../libvmaf/src/feature/cuda/integer_psnr_cuda.c).

### Surface

```c
#include "cuda/kernel_template.h"

typedef struct VmafCudaKernelLifecycle {
    CUstream str;       /* private non-blocking stream for readback */
    CUevent  submit;    /* recorded post-launch on picture stream    */
    CUevent  finished;  /* recorded post-readback on str             */
} VmafCudaKernelLifecycle;

typedef struct VmafCudaKernelReadback {
    VmafCudaBuffer *device;       /* device-side accumulator     */
    void           *host_pinned;  /* pinned host readback slot   */
    size_t          bytes;
} VmafCudaKernelReadback;

int  vmaf_cuda_kernel_lifecycle_init(VmafCudaKernelLifecycle *,
                                     VmafCudaState *);

int  vmaf_cuda_kernel_readback_alloc(VmafCudaKernelReadback *,
                                     VmafCudaState *, size_t bytes);

int  vmaf_cuda_kernel_submit_pre_launch(VmafCudaKernelLifecycle *,
                                        VmafCudaState *,
                                        VmafCudaKernelReadback *,
                                        CUstream picture_stream,
                                        CUevent dist_ready_event);

int  vmaf_cuda_kernel_collect_wait(VmafCudaKernelLifecycle *,
                                   VmafCudaState *);

int  vmaf_cuda_kernel_lifecycle_close(VmafCudaKernelLifecycle *,
                                      VmafCudaState *);

int  vmaf_cuda_kernel_readback_free(VmafCudaKernelReadback *,
                                    VmafCudaState *);
```

### What each helper covers

| Helper                                  | Boilerplate it replaces                                                       |
|-----------------------------------------|-------------------------------------------------------------------------------|
| `vmaf_cuda_kernel_lifecycle_init`       | `cuCtxPushCurrent` → `cuStreamCreateWithPriority` → 2× `cuEventCreate` → pop. |
| `vmaf_cuda_kernel_readback_alloc`       | `vmaf_cuda_buffer_alloc` + `vmaf_cuda_buffer_host_alloc` pair.                |
| `vmaf_cuda_kernel_submit_pre_launch`    | `cuMemsetD8Async` zero-out + `cuStreamWaitEvent` on dist's ready event.       |
| `vmaf_cuda_kernel_collect_wait`         | `cuStreamSynchronize` on the private stream.                                  |
| `vmaf_cuda_kernel_lifecycle_close`      | Stream sync + destroy + 2× event destroy, with partial-init safety.           |
| `vmaf_cuda_kernel_readback_free`        | Device-buffer free + struct-handle free; host-pinned free stays explicit.    |

### What stays in the kernel TU

- The per-metric `cuLaunchKernel(...)` call (grid dims, kernel parameter pack,
  function handle).
- The `cuModuleLoadData` / `cuModuleGetFunction` chain — kernel binary names
  and symbol counts vary per metric.
- The host-side reduction and score emission. PSNR's `10 * log10(peak² / mse)`
  is one line; `ssimulacra2` has a 6-band pyramid pool. Neither belongs in a
  shared header.
- The pinned-host buffer free (existing `vmaf_cuda_buffer_host_free`-class
  helper); the template intentionally does not reach across that boundary.

### Migration sketch

```c
typedef struct PsnrStateCuda {
    VmafCudaKernelLifecycle  lc;
    VmafCudaKernelReadback   sse;
    /* metric-specific: kernel handles, max constants, dict... */
} PsnrStateCuda;

static int init_fex_cuda(VmafFeatureExtractor *fex, ...)
{
    PsnrStateCuda *s = fex->priv;
    int err = vmaf_cuda_kernel_lifecycle_init(&s->lc, fex->cu_state);
    if (err) return err;
    err = vmaf_cuda_kernel_readback_alloc(&s->sse, fex->cu_state,
                                          sizeof(uint64_t));
    if (err) return err;
    /* metric-specific: module load + function resolve, peak constants */
    return 0;
}
```

The before/after diff for `integer_psnr_cuda.c` is roughly **−6 LOC of
host-side scaffolding** per kernel — small, but the win is mostly in the
shared error-handling and partial-init unwind paths, not the line count.

## Vulkan template

The Vulkan template captures the descriptor-pool + pipeline + per-WG int64
partials shape every Vulkan SSBO-only reduction kernel uses today. The
reference implementation is
[`psnr_vulkan.c`](../../libvmaf/src/feature/vulkan/psnr_vulkan.c).

### Surface

```c
#include "vulkan/kernel_template.h"

typedef struct VmafVulkanKernelPipeline {
    VkDescriptorSetLayout dsl;
    VkPipelineLayout      pipeline_layout;
    VkShaderModule        shader;
    VkPipeline            pipeline;
    VkDescriptorPool      desc_pool;
} VmafVulkanKernelPipeline;

typedef struct VmafVulkanKernelSubmit {
    VkCommandBuffer cmd;
    VkFence         fence;
} VmafVulkanKernelSubmit;

typedef struct VmafVulkanKernelPipelineDesc {
    uint32_t                       ssbo_binding_count;
    uint32_t                       push_constant_size;
    const uint32_t                *spv_bytes;
    size_t                         spv_size;
    VkComputePipelineCreateInfo    pipeline_create_info;
    uint32_t                       max_descriptor_sets;
} VmafVulkanKernelPipelineDesc;

int  vmaf_vulkan_kernel_pipeline_create(VmafVulkanContext *,
                                        const VmafVulkanKernelPipelineDesc *,
                                        VmafVulkanKernelPipeline *);

int  vmaf_vulkan_kernel_submit_begin(VmafVulkanContext *,
                                     VmafVulkanKernelSubmit *);

int  vmaf_vulkan_kernel_submit_end_and_wait(VmafVulkanContext *,
                                            VmafVulkanKernelSubmit *);

void vmaf_vulkan_kernel_submit_free(VmafVulkanContext *,
                                    VmafVulkanKernelSubmit *);

void vmaf_vulkan_kernel_pipeline_destroy(VmafVulkanContext *,
                                         VmafVulkanKernelPipeline *);
```

### What each helper covers

| Helper                                          | Boilerplate it replaces                                                      |
|-------------------------------------------------|------------------------------------------------------------------------------|
| `vmaf_vulkan_kernel_pipeline_create`            | DSL + pipeline layout + shader module + compute pipeline + descriptor pool. |
| `vmaf_vulkan_kernel_submit_begin`               | Allocate cmd buffer + begin recording + create fence (with rollback).        |
| `vmaf_vulkan_kernel_submit_end_and_wait`        | End recording + queue submit + fence wait.                                   |
| `vmaf_vulkan_kernel_submit_free`                | Destroy fence + free cmd buffer (partial-init safe).                         |
| `vmaf_vulkan_kernel_pipeline_destroy`           | `vkDeviceWaitIdle` + reverse-order destroy of the five pipeline objects.    |

### What stays in the kernel TU

- The shader bytecode header (`<feature>_spv.h`) — generated per-kernel by
  the `subdir('vulkan')` glslc chain.
- The push-constant struct layout. `PsnrPushConsts` and a hypothetical
  `Ssim4VifPushConsts` have nothing in common.
- Spec-constant population — the caller fills
  `pipeline_create_info.stage.pSpecializationInfo` before calling
  `vmaf_vulkan_kernel_pipeline_create`.
- Per-frame buffer alloc, host upload, descriptor-set allocation +
  binding-write, dispatch grid math, host-side reduction. These shapes
  diverge enough between kernels that a unified API would be either
  too narrow (just PSNR's shape) or too generic (callbacks for
  everything).

### Migration sketch

```c
typedef struct PsnrVulkanState {
    VmafVulkanContext        *ctx;
    int                       owns_ctx;
    VmafVulkanKernelPipeline  pl;
    /* metric-specific: per-plane buffers, push-const cache, ... */
} PsnrVulkanState;

static int init(VmafFeatureExtractor *fex, ...)
{
    PsnrVulkanState *s = fex->priv;
    /* ... resolve s->ctx ... */
    VmafVulkanKernelPipelineDesc desc = {
        .ssbo_binding_count = 3,
        .push_constant_size = sizeof(PsnrPushConsts),
        .spv_bytes          = psnr_spv,
        .spv_size           = psnr_spv_size,
        .max_descriptor_sets = 12,
        /* caller fills stage.pName + spec_info on pipeline_create_info */
    };
    desc.pipeline_create_info.stage.pName = "main";
    desc.pipeline_create_info.stage.pSpecializationInfo = &spec_info;
    return vmaf_vulkan_kernel_pipeline_create(s->ctx, &desc, &s->pl);
}

static int extract(VmafFeatureExtractor *fex, ...)
{
    PsnrVulkanState *s = fex->priv;
    VmafVulkanKernelSubmit sub;
    int err = vmaf_vulkan_kernel_submit_begin(s->ctx, &sub);
    if (err) return err;

    /* metric-specific: allocate descriptor sets, write bindings,
     * record commands on sub.cmd, etc. */

    err = vmaf_vulkan_kernel_submit_end_and_wait(s->ctx, &sub);
    /* host-side reduce + score emit */
    vmaf_vulkan_kernel_submit_free(s->ctx, &sub);
    return err;
}

static int close_fex(VmafFeatureExtractor *fex)
{
    PsnrVulkanState *s = fex->priv;
    vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl);
    /* metric-specific frees */
    return 0;
}
```

The before/after diff for `psnr_vulkan.c` is roughly **−30 LOC** — the
five vkCreate/vkDestroy pairs collapse into two helper calls each, and
the cleanup `goto`-ladder loses two labels.

## Migrating an existing kernel

Each kernel migration is its own PR, gated by:

1. **Netflix golden** (CPU only, untouched — the kernel doesn't run on the
   CPU lane).
2. **`/cross-backend-diff`** at `places=4` against the CPU reference, on
   every Netflix golden YUV pair the kernel is registered against (per
   [ADR-0214](../adr/0214-gpu-parity-ci-gate.md)).
3. The repo's standard `make lint` clean on every touched file
   (per [CLAUDE.md §12 r12](../../CLAUDE.md)).

Recommended migration order (smallest blast radius first):

| Backend | Kernel              | Reason                                                |
|---------|---------------------|-------------------------------------------------------|
| CUDA    | `integer_psnr_cuda` | Reference implementation; smallest delta.            |
| CUDA    | `ssimulacra2_cuda`  | Multi-readback pattern stresses the readback API.    |
| Vulkan  | `psnr_vulkan`       | Reference implementation; smallest delta.            |
| Vulkan  | `motion_vulkan`     | Single-dispatch + sub-group reduction.               |
| Vulkan  | `ssim_vulkan`       | Two-pass pipeline tests `pipeline_create` flexibility.|
| Vulkan  | `cambi_vulkan`      | Multi-pipeline (PASS=0/1/2 spec const).              |

Migrations are tracked as `T7-XX-followup-{a,b,c}` in `CHANGELOG.md`.

## Why per-backend (not cross-backend)

Sister-agent's GPU-template scope analysis (referenced by ADR-0246)
established that CUDA's async-stream + event model and Vulkan's
command-buffer + fence + descriptor-pool model share **no concrete
shape**. A cross-backend abstraction would force a lowest-common-denominator
API that captures neither well. The per-backend split keeps each header
honest about the platform it targets.

## Why helper functions (not macros)

Both templates use plain `static inline` functions, not preprocessor macros.
The trade-offs:

- **Debug stepping**: `cuda-gdb` / Nsight / RenderDoc / vkconfig step through
  inline functions; macros expand to a single compound statement that
  shows up as one line in the source view.
- **Type-checking**: missing parameters or wrong-type pointers produce
  compiler errors at the helper site, not at some inscrutable point inside
  a macro expansion.
- **The macros that do pay off live elsewhere**:
  [`cuda_helper.cuh`](../../libvmaf/src/cuda/cuda_helper.cuh) provides
  `CHECK_CUDA_GOTO` / `CHECK_CUDA_RETURN`, which are macros precisely because
  their `goto label` form needs textual substitution. The kernel-template
  helpers use those macros internally.

## See also

- [ADR-0246](../adr/0246-gpu-kernel-template.md) — design decision and
  alternatives.
- [`libvmaf/src/cuda/AGENTS.md`](../../libvmaf/src/cuda/AGENTS.md) — kernel
  template invariant row.
- [`libvmaf/src/vulkan/AGENTS.md`](../../libvmaf/src/vulkan/AGENTS.md) —
  kernel template invariant row.
- [`docs/backends/cuda/overview.md`](cuda/overview.md) — broader CUDA
  backend overview.
- [`docs/backends/vulkan/overview.md`](vulkan/overview.md) — broader Vulkan
  backend overview.
