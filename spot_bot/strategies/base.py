from dataclasses import dataclass
from typing import Any, Dict, Protocol

import pandas as pd


@dataclass(frozen=True)
class Intent:
    desired_exposure: float  # 0..1, fraction of equity in asset (BTC)
    reason: str
    diagnostics: Dict[str, Any]


class Strategy(Protocol):
    def generate_intent(self, features_df: pd.DataFrame) -> Intent: ...
