Convert three stale TODO markers to structured Deferred annotations and implement
the `_Runner` ABC contract: `measure_quant_drop_per_ep._Runner` now inherits
`abc.ABC` with `@abc.abstractmethod` on `infer` (closes #842 gap);
`vmaf_cuda_fetch_preallocated_picture` HOST-pool TODO replaced with a Deferred
note naming the prerequisite (separate host-callback pool, T9.x backlog);
`adm_dwt2` x-mirroring TODO replaced with a Deferred note citing the tile-overlap
audit needed before the abs() refactor can land safely.
