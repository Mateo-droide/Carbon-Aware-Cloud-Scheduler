# src/schedulers/least_loaded.py
from typing import Any, Dict
from src.schedulers.base import BaseScheduler
from src.models.schemas import Workload

class LeastLoadedScheduler(BaseScheduler):
    """
    Planificador que selecciona la región con el menor tiempo de espera estimado (carga).
    En caso de empate (ej. múltiples regiones con espera 0), selecciona la región más económica.
    """
    def __init__(self, simulator: Any, num_regions: int = 3):
        self.sim = simulator
        self.num_regions = num_regions

    def select_action(self, obs: Any = None, info: Dict[str, Any] = None) -> int:
        # Extraemos la tarea entrante enviada en el parámetro obs
        workload: Workload = obs
        
        best_region = -1
        min_wait_time = float('inf')
        min_cost = float('inf')

        for region_id in range(self.num_regions):
            # 1. Consultar el simulador para estimar la carga (tiempo de espera)
            wait_time = self.sim.estimate_wait_time(
                region_id=region_id, 
                arrival_h=workload.arrival_time_h, 
                cpu_req=workload.cpu_req
            )
            
            # 2. Calcular costo operativo como criterio de desempate
            region_cfg = self.sim.regions[region_id].config
            cost = workload.cpu_req * workload.duration_h * region_cfg.price_per_vcpu_h

            # 3. Selección de la mejor región
            if wait_time < min_wait_time:
                min_wait_time = wait_time
                min_cost = cost
                best_region = region_id
            elif wait_time == min_wait_time:
                # 4. Resolver empates de forma determinista por costo
                if cost < min_cost:
                    min_cost = cost
                    best_region = region_id
                    
        return best_region

    def reset(self) -> None:
        """No requiere estado interno entre episodios, pero cumple la interfaz."""
        pass