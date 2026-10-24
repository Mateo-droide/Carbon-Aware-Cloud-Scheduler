# src/schedulers/q_learning.py
import os
from typing import Any, Dict, Optional
import numpy as np

from src.models.schemas import Workload
from src.schedulers.base import BaseScheduler


class TabularQLearningAgent(BaseScheduler):
    """
    Agente Q-Learning Tabular implementado desde cero con NumPy (V1).
    Actualización de Bellman:
    Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
    """

    def __init__(
        self,
        num_states: int = 1458,
        num_actions: int = 3,
        alpha: float = 0.15,
        gamma: float = 0.90,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.992,
        seed: int = 42,
        training: bool = True,
        simulator: Optional[Any] = None,
        discretizer: Optional[Any] = None,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start if training else 0.0
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.training = training
        self.rng = np.random.default_rng(seed)

        # Referencias opcionales para evaluación directa con select_action(obs=workload)
        self.simulator = simulator
        self.discretizer = discretizer

        # Tabla Q de dimensiones (1458, 3) inicializada en ceros
        self.q_table = np.zeros((num_states, num_actions), dtype=np.float64)

    def _resolve_state_idx(self, obs: Any) -> int:
        """Permite recibir un índice entero de estado o un objeto Workload directamente."""
        if isinstance(obs, (int, np.integer)):
            return int(obs)
        if isinstance(obs, Workload):
            if self.simulator is None or self.discretizer is None:
                raise ValueError(
                    "Para pasar obs=Workload a TabularQLearningAgent, inyecta simulator y discretizer."
                )
            return self.discretizer.discretize_from_simulator(self.simulator, obs)
        raise TypeError(f"Tipo de observación no soportado en Q-Learning: {type(obs)}")

    def select_action(self, obs: Any, info: Dict[str, Any] = None) -> int:
        state_idx = self._resolve_state_idx(obs)

        # Exploración epsilon-greedy únicamente en modo entrenamiento
        if self.training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.num_actions))

        q_values = self.q_table[state_idx]

        # Mitigación Riesgo 3: Si el estado jamás fue visitado (todos los Q == 0.0),
        # desempatar hacia la región con menor espera proyectada si el simulador está disponible
        if np.allclose(q_values, q_values[0]):
            if isinstance(obs, Workload) and self.simulator is not None:
                waits = [
                    self.simulator.estimate_wait_time(r, obs.arrival_time_h, obs.cpu_req)
                    for r in range(self.num_actions)
                ]
                return int(np.argmin(waits))
            return int(self.rng.integers(0, self.num_actions))

        return int(np.argmax(q_values))

    def update(self, state: int, action: int, reward: float, next_state: int, terminated: bool) -> float:
        """Aplica la ecuación de Bellman y devuelve el error TD (Temporal Difference)."""
        if not self.training:
            return 0.0

        current_q = self.q_table[state, action]
        max_next_q = 0.0 if terminated else float(np.max(self.q_table[next_state]))
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q

        self.q_table[state, action] = current_q + self.alpha * td_error
        return float(td_error)

    def decay_epsilon(self) -> None:
        """Reduce epsilon al finalizar cada episodio de entrenamiento."""
        if self.training:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def set_eval_mode(self) -> None:
        """Congela la política (sin exploración ni actualización de pesos)."""
        self.training = False
        self.epsilon = 0.0

    def save(self, filepath: str = "models_saved/q_table_v1.npy") -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.save(filepath, self.q_table)

    def load(self, filepath: str = "models_saved/q_table_v1.npy") -> None:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No se encontró la política entrenada en: {filepath}")
        self.q_table = np.load(filepath)
        self.set_eval_mode()

    def reset(self) -> None:
        pass