# AVX-512 / SIMD Best Practices Research

> Sources: Agner Fog "Optimizing Assembly" (156pp), Algorithmica SIMD chapters, AMD/Intel docs
> Date: 2026-02-19

## Key Findings for libvmaf CPU Backend

### Register Types & Widths
| Register | Width | int32 elements | int16 elements | Usage |
|----------|-------|----------------|----------------|-------|
| XMM      | 128-bit | 4 | 8 | SSE, always available |
| YMM      | 256-bit | 8 | 16 | AVX2, good compatibility |
| ZMM      | 512-bit | 16 | 32 | AVX-512, highest throughput |

### AVX-512 Mask Registers (Critical Feature)
- **8 mask registers** (k0-k7): each bit controls one vector element
- `vpcmpud k1, ymm0, ymm1, 5` — comparison directly to mask register
- **Masked load**: `vmovdqu32 zmm1{k1}{z}, [mem]` — zeroing-masking ({z}) avoids false dependencies
- **Masked store**: `vmovdqu32 [mem]{k1}, zmm1` — only stores elements where mask bit = 1
- **Masked compute**: `vpaddd zmm1{k1}, zmm2, zmm3` — only processes active elements
- "No cost to adding a mask to a 512-bit vector instruction" (Agner Fog)
- **Loop tail handling**: Instead of scalar cleanup loop, use mask register to process remaining elements

### Warm-Up Latency (IMPORTANT for benchmarks)
- **10-20 µs** to power up 256/512-bit execution units after idle period
- During warm-up, 256-bit ops are done as 2×128-bit → lower throughput
- Upper 256/512-bit units turn off after ~**1 ms** of no large vector instructions
- **Workaround**: Execute a dummy 256/512-bit instruction to warm up in advance
- **AMD Zen 5 specifics**: AVX-512 runs at full frequency (NO downclocking unlike Intel Skylake-X)

### Loop Patterns
```
; AVX-512 DAXPY pattern (Agner Fog section 12.10)
vbroadcastsd zmm2, [DA]             ; broadcast scalar to all elements
.loop:
    vmovupd zmm1, [rdi + rax]       ; load 8 doubles
    vfnmadd231pd zmm1, zmm2, [rsi + rax]  ; fused multiply-add
    vmovupd [rdi + rax], zmm1       ; store result
    add rax, 64                     ; advance by 64 bytes (8 doubles)
    cmp rax, rcx
    jb .loop
; Measured: 1.5 cycles/iteration on Skylake (bottleneck = L1 cache throughput)
```

### Loop Unrolling Guidance
- **Don't over-unroll**: Larger loop body → µop cache pressure → evicts other code
- AVX-512 instructions are longer (EVEX prefix = 4 bytes) → code bloat faster
- Unrolling 2× is usually sufficient; 4× rarely helps on modern CPUs
- Use `__builtin_expect` or PGO for branch prediction hints

### GCC Vector Extensions (Alternative to Intrinsics)
```c
typedef int v16si __attribute__((vector_size(64)));  // 16 × int32
v16si a = *(v16si*)ptr_a;
v16si b = *(v16si*)ptr_b;
v16si c = a + b;  // Uses natural operators
*(v16si*)ptr_c = c;
```
- Pro: Portable, readable, compiler auto-selects instructions
- Con: Less control over specific instruction selection
- Can mix with intrinsics when needed

### Memory Alignment
- **64-byte alignment** for ZMM (AVX-512): `__attribute__((aligned(64)))`
- `vmovdqa64` (aligned) vs `vmovdqu64` (unaligned) — modern CPUs: unaligned penalty is small when naturally aligned
- Allocate with `aligned_alloc(64, size)` or `posix_memalign(&ptr, 64, size)`

### AMD Zen 5 Specifics
- Full 512-bit execution units (not split like Zen 4 which was 2×256-bit)
- AVX-512 runs without frequency throttling
- L1D: 48 KB, L1I: 32 KB, L2: 1 MB per core, L3: 64 MB (shared)
- 3D V-Cache variant: 128 MB L3 (Ryzen 9 9950X3D)
- 6 ALU ports, 3 load/store ports per core
- Branch prediction: 2× deeper tables vs Zen 4

## Applicability to libvmaf

### Current State Assessment Needed
- Check existing AVX-512 kernels (VIF, ADM, Motion) for:
  - Are they using ZMM registers or falling back to YMM?
  - Mask register usage for loop tails?
  - Memory alignment of input/output buffers?
  - Loop unrolling factor?

### Potential New Optimizations
1. **Warm-up instruction** — add a dummy ZMM op early in processing to avoid 10-20µs penalty on first frame
2. **Mask-based loop tails** — replace scalar cleanup loops with masked ZMM operations
3. **64-byte aligned allocations** — ensure picture buffers are 64-byte aligned for AVX-512
4. **Full ZMM width** — verify all SIMD paths use 512-bit where available on Zen 5
5. **Fused multiply-add** — `vfmadd231ps` for mul+add patterns (single instruction, better throughput)
6. **Reduce unrolling** — if loops are unrolled 4×+, try 2× and measure
7. **GCC vector extensions** for new code — more maintainable than raw intrinsics

### GPU Compute Occupancy (Cross-Reference)
From GPUOpen research on AMD GCN/RDNA architecture:
- VGPR budget: 32 VGPRs for max occupancy (2 thread groups per CU)
- LDS for neighborhood processing: 32×32 group = only 13% border sample overhead
- Scalar data in SGPRs: zero VGPR cost for uniform values
- Bit-packing: store multiple small values in single VGPR to save register pressure
- Applies to GPU compute shaders on AMD iGPU (RADV)
