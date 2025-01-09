from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RdPoint:
    rate: float
    metric: float

    def __hash__(self) -> int:
        return hash((self.rate, self.metric))
