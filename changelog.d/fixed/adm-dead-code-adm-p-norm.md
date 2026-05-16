**fix(adm):** Remove stale dead-code block and replace the `adm_p_norm` TODO
comment in `integer_adm.c` with a brief explanatory note. The exponent is fixed
at 3.0f per the Netflix training-data contract; no runtime parameterisation is
planned until a model retrain occurs.
