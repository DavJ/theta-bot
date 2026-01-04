from dataclasses import dataclass
from typing import Dict, Literal


@dataclass(frozen=True)
class RegimeDecision:
    risk_state: Literal["ON", "REDUCE", "OFF"]
    risk_budget: float  # float in [0,1]
    reason: str
    diagnostics: Dict
