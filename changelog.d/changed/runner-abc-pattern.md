- Converted `_Runner` in `ai/scripts/measure_quant_drop_per_ep.py` from a bare
  class with a manual `raise NotImplementedError` stub to an `abc.ABC` subclass
  with `@abc.abstractmethod` on `infer()`, closing issue #842. Concrete
  subclasses (`_OrtRunner`, `_OpenVinoRunner`) already implement `infer()`
  unchanged; the refactor simply enforces the contract at instantiation time
  rather than at call time.
