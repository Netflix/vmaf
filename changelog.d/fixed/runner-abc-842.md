`_Runner` in `ai/scripts/measure_quant_drop_per_ep.py` is now an `abc.ABC`
with `@abc.abstractmethod` on `infer`, so missing implementations raise
`TypeError` at instantiation time rather than at the call site (closes #842).
