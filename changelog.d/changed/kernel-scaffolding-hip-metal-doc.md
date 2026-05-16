Extend `docs/backends/kernel-scaffolding.md` with HIP and Metal kernel
lifecycle template sections and a shared four-phase lifecycle contract
(init / submit / collect / close), eliminating duplicate contract prose
from `hip/kernel_template.h` and `metal/kernel_template.h` (ADR-0484).
