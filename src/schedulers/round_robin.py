# src/schedulers/round_robin.py
from typing import Any, Dict
from src.schedulers.base import BaseScheduler

class RoundRobinScheduler(BaseScheduler):
    """
    Baseline determinista que asigna cargas cíclicamente:
    a_t = (i mod 3) -> Cloud A (0) -> Cloud B (1) -> Cloud C (2).
    """

    def __init__(self, num_regions: int = 3):
        self.num_regions = num_regions
        self.step_counter = 0

    def select_action(self, obs: Any = None, info: Dict[str, Any] = None) -> int:
        region_id = self.step_counter % self.num_regions
        self.step_counter += 1
        return region_id

    def reset(self) -> None:
        self.step_counter = 0