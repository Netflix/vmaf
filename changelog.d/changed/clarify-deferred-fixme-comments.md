Convert two stale `//^FIXME:` markers in the CUDA picture-unref path and one
`// TODO:` marker in `integer_adm.c` into properly annotated `/* Deferred */`
comments.  No behaviour change; the deferred work (picture-callback ABI and
`adm_p_norm` parameterisation) still requires future model retraining or a
stable callback ABI.
