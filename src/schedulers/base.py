# src/schedulers/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseScheduler(ABC):
    """Interfaz común para Round Robin (V1/V2), Q-Learning (V1) y PPO/Pareto (V2)."""

    @abstractmethod
    def select_action(self, obs: Any, info: Dict[str, Any] = None) -> int:
        """Devuelve el ID de región seleccionada: 0 (Cloud A), 1 (Cloud B) o 2 (Cloud C)."""
        pass

    def reset(self) -> None:
        """Reinicia contadores internos al comenzar un nuevo episodio."""
        pass