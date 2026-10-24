# src/models/schemas.py
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass(frozen=True)
class Workload:
    """Representa una tarea entrante w_i = (t_i, c_i, d_i, prio_i, DDL_i)."""
    task_id: int
    arrival_time_h: float      # t_i: Instante de llegada (horas)
    cpu_req: int               # c_i: vCPUs requeridas {2, 4, 8, 16}
    duration_h: float          # d_i: Duración de ejecución en horas [0.25, 6.0]
    priority: int              # prio_i: Prioridad {1: Baja, 2: Media, 3: Alta}
    slack_h: float             # \Delta_i: Holgura inicial antes del deadline (horas)
    deadline_h: float          # DDL_i = t_i + d_i + \Delta_i (horas)

@dataclass
class RegionConfig:
    """Parámetros estáticos de una región cloud leídos desde YAML."""
    region_id: int
    name: str
    ci_mean_g_kwh: float
    ci_amp_g_kwh: float
    ci_phase_rad: float
    price_per_vcpu_h: float
    max_vcpu: int
    net_latency_h: float
    power_kw_per_vcpu: float

@dataclass
class RegionRuntimeState:
    """
    Estado dinámico de una región para gestionar capacidad y cola FIFO.
    active_allocations almacena tuplas (end_time_h, cpu_allocated) de tareas en curso.
    """
    config: RegionConfig
    active_allocations: List[Tuple[float, int]] = field(default_factory=list)

@dataclass(frozen=True)
class TaskMetrics:
    """Resultado físico y operativo tras ejecutar la tarea i en la región elegida."""
    task_id: int
    assigned_region: int
    arrival_time_h: float
    start_exec_time_h: float   # t_{i,start}: Instante real en que obtiene CPU
    Wait_time_h: float         # D_{i,wait} = t_{i,start} - t_i (espera en cola FIFO)
    net_latency_h: float       # L_{r,net}
    total_response_h: float    # D_{i,real} = d_i + L_{r,net} + D_{i,wait}
    completion_time_h: float   # C_i = t_i + D_{i,real}
    deadline_h: float          # DDL_i
    carbon_intensity_g_kwh: float  # CI_r(t_{i,start})
    energy_kwh: float          # E_elec = c_i * P_{vCPU,r} * d_i
    co2_emissions_g: float     # E_i = E_elec * CI_r(t_{i,start})
    cost_usd: float            # Cost_i = c_i * d_i * p_r
    sla_violated: int          # v_i \in {0, 1}: 1 si C_i > DDL_i, 0 en caso contrario