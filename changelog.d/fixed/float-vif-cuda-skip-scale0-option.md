**float_vif CUDA: wire missing `vif_skip_scale0` option** — `float_vif_cuda`
implemented `vif_skip_scale0` suppression logic in `collect_fex_cuda` and the
debug path, but never registered the option in its `VmafOption` table. The
field was always zero-initialized (`false`), silently ignoring any caller
setting the option. The option is now registered (name `vif_skip_scale0`,
alias `ssclz`, matching `float_vif.c`), so scale-0 suppression works as
intended on the CUDA backend.
