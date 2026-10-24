# src/evaluation/runner.py
import os
from dataclasses import asdict
from typing import Dict, List, Tuple
import yaml
import numpy as np
import pandas as pd

from src.models.workload import WorkloadGenerator
from src.env.cloud_simulator import CloudSimulator
from src.env.gym_env import CarbonCloudEnv, DiscretizerWrapper
from src.schedulers.round_robin import RoundRobinScheduler
from src.schedulers.least_loaded import LeastLoadedScheduler
from src.schedulers.q_learning import TabularQLearningAgent


class ExperimentRunner:
    """
    Ejecutor experimental reproducible (Entregables R3, R6, R7, R8).
    Evalúa múltiples semillas de prueba disjuntas del entrenamiento (200..209).
    """

    def __init__(
        self,
        regions_cfg_path: str = "configs/regions.yaml",
        rl_cfg_path: str = "configs/rl_config.yaml",
        model_path: str = "models_saved/q_table_v1.npy",
        test_seeds: List[int] = None,
    ):
        self.regions_cfg_path = regions_cfg_path
        self.rl_cfg_path = rl_cfg_path
        self.model_path = model_path
        self.test_seeds = test_seeds if test_seeds is not None else list(range(200, 210))

        with open(regions_cfg_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        base_env = CarbonCloudEnv(
            regions_cfg_path=regions_cfg_path,
            rl_cfg_path=rl_cfg_path,
        )
        self.discretizer = DiscretizerWrapper(base_env)

    def export_sample_workloads(self, seed: int = 200) -> None:
        """Exporta las trazas de workloads en CSV para cumplir con el entregable R3."""
        out_dir = "data/generated_workloads"
        os.makedirs(out_dir, exist_ok=True)

        for scen_name, scen_params in self.cfg["scenarios"].items():
            gen = WorkloadGenerator(seed=seed)
            trace = gen.generate_trace(
                num_tasks=scen_params["num_tasks"],
                arrival_rate_per_h=scen_params["arrival_rate_per_h"],
            )
            df_trace = pd.DataFrame([asdict(w) for w in trace])
            df_trace.to_csv(f"{out_dir}/workloads_{scen_name}_seed{seed}.csv", index=False)

    def run_all(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Ejecuta los 3 escenarios x 3 algoritmos x N semillas.
        Devuelve (df_raw_runs, df_summary_stats).
        """
        self.export_sample_workloads(seed=self.test_seeds[0])
        raw_records = []

        for scen_name, scen_params in self.cfg["scenarios"].items():
            for seed in self.test_seeds:
                gen = WorkloadGenerator(seed=seed)
                workloads = gen.generate_trace(
                    num_tasks=scen_params["num_tasks"],
                    arrival_rate_per_h=scen_params["arrival_rate_per_h"],
                )

                sim = CloudSimulator(
                    config_path=self.regions_cfg_path,
                    scenario_name=scen_name,
                    seed=seed,
                )

                q_agent = TabularQLearningAgent(
                    training=False,
                    simulator=sim,
                    discretizer=self.discretizer,
                )
                q_agent.load(self.model_path)

                schedulers = {
                    "Round Robin": RoundRobinScheduler(num_regions=3),
                    "Least Loaded": LeastLoadedScheduler(sim),
                    "Q-Learning (V1)": q_agent,
                }

                for sched_name, scheduler in schedulers.items():
                    sim.reset(seed=seed)
                    scheduler.reset()

                    region_counts = {0: 0, 1: 0, 2: 0}
                    for w in workloads:
                        action = scheduler.select_action(obs=w)
                        region_counts[int(action)] += 1
                        sim.step_task(w, action)

                    summary = sim.summarize_episode()
                    raw_records.append({
                        "scenario": scen_name,
                        "scheduler": sched_name,
                        "seed": seed,
                        "total_co2_g": summary["total_co2_g"],
                        "total_cost_usd": summary["total_cost_usd"],
                        "mean_response_time_h": summary["mean_response_time_h"],
                        "mean_latency_overhead_h": summary["mean_latency_overhead_h"],
                        "sla_violations_count": summary["sla_violations_count"],
                        "sla_violation_rate_pct": summary["sla_violation_rate_pct"],
                        "pct_cloud_A": (region_counts[0] / len(workloads)) * 100.0,
                        "pct_cloud_B": (region_counts[1] / len(workloads)) * 100.0,
                        "pct_cloud_C": (region_counts[2] / len(workloads)) * 100.0,
                    })

        df_raw = pd.DataFrame(raw_records)

        # Agregar media y desviación estándar por (scenario, scheduler)
        metrics_cols = [
            "total_co2_g",
            "total_cost_usd",
            "mean_response_time_h",
            "mean_latency_overhead_h",
            "sla_violation_rate_pct",
            "pct_cloud_A",
            "pct_cloud_B",
            "pct_cloud_C",
        ]
        df_summary = (
            df_raw.groupby(["scenario", "scheduler"], sort=False)[metrics_cols]
            .agg(["mean", "std"])
            .reset_index()
        )
        # Aplanar nombres de columnas multi-índice
        df_summary.columns = [
            f"{col[0]}_{col[1]}" if col[1] else col[0] for col in df_summary.columns
        ]

        os.makedirs("results", exist_ok=True)
        df_raw.to_csv("results/eval_runs_detailed.csv", index=False)
        df_summary.to_csv("results/eval_summary_table.csv", index=False)

        return df_raw, df_summary