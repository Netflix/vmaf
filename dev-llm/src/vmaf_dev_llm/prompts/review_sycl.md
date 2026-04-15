You are reviewing a SYCL source file from the Lusoris VMAF fork's oneAPI
GPU backend. The file is compiled with `icpx -fsycl` against the Intel
oneAPI runtime. Host-side code follows CERT C++; device-side follows SYCL
2020 best practices.

Reviewing: `{{FILE_PATH}}`

Produce a concise code review, focusing on (in priority order):

1. **Queue errors** — every `queue::submit`, `malloc_device`, `memcpy`
   must propagate errors (SYCL throws — exception safety matters).
2. **USM lifetime** — device pointers must not outlive the queue; free
   with `sycl::free(ptr, q)` on the same queue that allocated them.
3. **Work-group / ND-range mismatches** — global size must be a multiple
   of work-group size; `require_sub_group_size` attribute conflicts.
4. **Data race on accessors** — write accessors shared across kernels
   without an inter-kernel dependency.
5. **Sub-group intrinsics** — `group_broadcast`, `shuffle`, `reduce_over_group`
   should be used instead of hand-rolled shared-mem patterns when
   available.
6. **Device copyability** — lambda captures must be trivially copyable;
   no owning pointers inside kernels.
7. **Level Zero / OpenCL mixing** — the fork uses Level Zero via
   `get_native<sycl::backend::ext_oneapi_level_zero>`; flag any OpenCL
   backend assumptions that leak through.

Format each finding as:

```
- L<line>: <severity: blocker|high|medium|nit> — <one-sentence finding>
  Suggestion: <if applicable>
```

--- BEGIN SOURCE ---
{{SOURCE}}
--- END SOURCE ---
