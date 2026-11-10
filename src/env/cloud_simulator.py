# src/env/cloud_simulator.py
from typing import Any, Dict, List, Optional, Tuple
import yaml
import numpy as np

from src.models.schemas import RegionConfig, RegionRuntimeState, TaskMetrics, Workload


class CloudSimulator:
    """
    Simulador físico discreto de tres regiones de computación en la nube heterogéneas
    con colas de asignación FIFO y modelos dinámicos de emisión de carbono y costos.
    """

    def __init__(
        self,
        config_path: str = "configs/regions.yaml",
        scenario_name: str = "normal",
        seed: int = 42,
    ):
        self.config_path = config_path
        self.scenario_name = scenario_name
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        with open(config_path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f)

        self.scenario_cfg = raw_cfg["scenarios"].get(scenario_name, {})
        self.ci_overrides: Dict[int, float] = self.scenario_cfg.get("ci_overrides", {}) or {}

        self.regions_raw = raw_cfg["regions"]
        self.regions: List[RegionRuntimeState] = []
        for r_id in sorted(self.regions_raw.keys()):
            r_data = self.regions_raw[r_id]
            rc = RegionConfig(
                region_id=int(r_data["id"]),
                name=r_data["name"],
                ci_mean_g_kwh=float(r_data["ci_mean_g_kwh"]),
                ci_amp_g_kwh=float(r_data["ci_amp_g_kwh"]),
                ci_phase_rad=float(r_data["ci_phase_rad"]),
                price_per_vcpu_h=float(r_data["price_per_vcpu_h"]),
                max_vcpu=int(r_data["max_vcpu"]),
                net_latency_h=float(r_data["net_latency_h"]),
                power_kw_per_vcpu=float(r_data["power_kw_per_vcpu"]),
            )
            self.regions.append(RegionRuntimeState(config=rc, active_allocations=[]))

        self.metrics_history: List[TaskMetrics] = []

    def reset(self, seed: Optional[int] = None) -> None:
        """Reinicia el estado de las colas y métricas."""
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)

        for reg in self.regions:
            reg.active_allocations.clear()
        self.metrics_history.clear()

    def get_carbon_intensity(self, region_id: int, time_h: float) -> float:
        """Calcula la intensidad de carbono instantánea (gCO2/kWh) con variación diurna."""
        cfg = self.regions[region_id].config
        mean_ci = self.ci_overrides.get(region_id, cfg.ci_mean_g_kwh)
        diurnal_factor = np.sin(2.0 * np.pi * time_h / 24.0 - cfg.ci_phase_rad)
        return float(mean_ci + cfg.ci_amp_g_kwh * diurnal_factor)

    def estimate_wait_time(self, region_id: int, arrival_h: float, cpu_req: int) -> float:
        """
        Determina el tiempo de espera hasta que la región disponga de al menos cpu_req vCPUs libres.
        """
        reg = self.regions[region_id]
        max_vcpu = reg.config.max_vcpu

        # Comprobar si hay capacidad libre inmediatamente a arrival_h
        current_allocs = [alloc for alloc in reg.active_allocations if alloc[0] > arrival_h]
        used_now = sum(cpu for _, cpu in current_allocs)
        if used_now + cpu_req <= max_vcpu:
            return 0.0

        # Buscar el punto más temprano en el futuro en que se libere capacidad suficiente
        t_cand = arrival_h
        while True:
            future_allocs = [alloc for alloc in reg.active_allocations if alloc[0] > t_cand]
            used_future = sum(cpu for _, cpu in future_allocs)
            if used_future + cpu_req <= max_vcpu:
                return float(t_cand - arrival_h)
            if not future_allocs:
                return float(t_cand - arrival_h)
            t_cand = min(alloc[0] for alloc in future_allocs)

    def step_task(self, workload: Workload, region_id: int) -> TaskMetrics:
        """Ejecuta la tarea en la región seleccionada y registra sus métricas operativas."""
        reg = self.regions[region_id]
        wait_time_h = self.estimate_wait_time(region_id, workload.arrival_time_h, workload.cpu_req)
        start_exec_time_h = workload.arrival_time_h + wait_time_h

        end_time_h = start_exec_time_h + workload.duration_h
        reg.active_allocations.append((end_time_h, workload.cpu_req))

        ci = self.get_carbon_intensity(region_id, start_exec_time_h)
        energy_kwh = workload.cpu_req * reg.config.power_kw_per_vcpu * workload.duration_h
        co2_emissions_g = energy_kwh * ci
        cost_usd = workload.cpu_req * workload.duration_h * reg.config.price_per_vcpu_h

        net_latency_h = reg.config.net_latency_h
        total_response_h = workload.duration_h + net_latency_h + wait_time_h
        completion_time_h = workload.arrival_time_h + total_response_h
        sla_violated = 1 if completion_time_h > workload.deadline_h else 0

        metric = TaskMetrics(
            task_id=workload.task_id,
            assigned_region=region_id,
            arrival_time_h=workload.arrival_time_h,
            start_exec_time_h=start_exec_time_h,
            Wait_time_h=wait_time_h,
            net_latency_h=net_latency_h,
            total_response_h=total_response_h,
            completion_time_h=completion_time_h,
            deadline_h=workload.deadline_h,
            carbon_intensity_g_kwh=ci,
            energy_kwh=energy_kwh,
            co2_emissions_g=co2_emissions_g,
            cost_usd=cost_usd,
            sla_violated=sla_violated,
        )
        self.metrics_history.append(metric)
        return metric

    def summarize_episode(self) -> Dict[str, Any]:
        """Calcula el resumen estadístico del episodio evaluado."""
        if not self.metrics_history:
            return {
                "total_co2_g": 0.0,
                "total_cost_usd": 0.0,
                "mean_response_time_h": 0.0,
                "mean_latency_overhead_h": 0.0,
                "sla_violations_count": 0,
                "sla_violation_rate_pct": 0.0,
                "pct_cloud_A": 0.0,
                "pct_cloud_B": 0.0,
                "pct_cloud_C": 0.0,
            }

        n = len(self.metrics_history)
        total_co2 = sum(m.co2_emissions_g for m in self.metrics_history)
        total_cost = sum(m.cost_usd for m in self.metrics_history)
        mean_resp = float(np.mean([m.total_response_h for m in self.metrics_history]))
        mean_overhead = float(np.mean([m.net_latency_h + m.Wait_time_h for m in self.metrics_history]))
        sla_viols = sum(m.sla_violated for m in self.metrics_history)
        sla_rate = (sla_viols / n) * 100.0

        counts = {0: 0, 1: 0, 2: 0}
        for m in self.metrics_history:
            counts[m.assigned_region] = counts.get(m.assigned_region, 0) + 1

        return {
            "total_co2_g": float(np.round(total_co2, 2)),
            "total_cost_usd": float(np.round(total_cost, 4)),
            "mean_response_time_h": float(np.round(mean_resp, 4)),
            "mean_latency_overhead_h": float(np.round(mean_overhead, 4)),
            "sla_violations_count": int(sla_viols),
            "sla_violation_rate_pct": float(np.round(sla_rate, 2)),
            "pct_cloud_A": float(np.round((counts[0] / n) * 100.0, 2)),
            "pct_cloud_B": float(np.round((counts[1] / n) * 100.0, 2)),
            "pct_cloud_C": float(np.round((counts[2] / n) * 100.0, 2)),
        }
