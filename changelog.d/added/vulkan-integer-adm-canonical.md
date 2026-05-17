Introduce `integer_adm_vulkan.c` as canonical Vulkan integer ADM extractor
with dedicated `integer_adm.comp` / `integer_adm_reduce.comp` GLSL shaders.
The extractor is registered as `"integer_adm_vulkan"` and mirrors the naming
of `integer_vif_vulkan.c` and the CUDA / SYCL twins.  The legacy
`adm_vulkan.c` is retained as a compatibility shim.  (ADR-0468.)
