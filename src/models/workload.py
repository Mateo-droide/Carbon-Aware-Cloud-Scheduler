# src/models/workload.py
from typing import List
import numpy as np
from src.models.schemas import Workload

class WorkloadGenerator:
    """Generador estocástico reproducible de cargas de trabajo cloud."""

    CPU_OPTIONS = [2, 4, 8, 16]
    PRIORITY_OPTIONS = [1, 2, 3]  # 1: Batch/Baja, 2: Normal, 3: Crítica/Alta

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_trace(self, num_tasks: int, arrival_rate_per_h: float) -> List[Workload]:
        """
        Genera una secuencia de N tareas ordenadas por tiempo de llegada t_i.
        """
        workloads: List[Workload] = []
        current_time_h = 0.0

        for task_id in range(num_tasks):
            # Tiempo entre llegadas exponencial (proceso de Poisson con tasa \lambda)
            inter_arrival_h = float(self.rng.exponential(scale=1.0 / arrival_rate_per_h))
            current_time_h += inter_arrival_h

            cpu_req = int(self.rng.choice(self.CPU_OPTIONS, p=[0.40, 0.35, 0.15, 0.10]))
            duration_h = float(np.round(self.rng.uniform(0.25, 4.0), 4))
            priority = int(self.rng.choice(self.PRIORITY_OPTIONS, p=[0.30, 0.50, 0.20]))

            # La holgura (\Delta_i) depende de la prioridad: tareas críticas tienen deadlines más estrictos
            if priority == 3:
                slack_h = float(np.round(self.rng.uniform(0.15, 0.60), 4))
            elif priority == 2:
                slack_h = float(np.round(self.rng.uniform(0.50, 1.50), 4))
            else:
                slack_h = float(np.round(self.rng.uniform(1.20, 3.50), 4))

            deadline_h = current_time_h + duration_h + slack_h

            workloads.append(
                Workload(
                    task_id=task_id,
                    arrival_time_h=current_time_h,
                    cpu_req=cpu_req,
                    duration_h=duration_h,
                    priority=priority,
                    slack_h=slack_h,
                    deadline_h=deadline_h,
                )
            )

        return workloads