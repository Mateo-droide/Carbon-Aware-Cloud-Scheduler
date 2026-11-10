# src/env/gym_env.py
from typing import Any, Dict, Optional, Tuple
import yaml
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.models.workload import WorkloadGenerator
from src.models.schemas import Workload
from src.env.cloud_simulator import CloudSimulator


class CarbonCloudEnv(gym.Env):
    """
    Entorno Gymnasium para asignación de tareas cloud consciente de carbono (V1).
    Proporciona un espacio de observación continuo [0, 1]^11 y recompensas multiobjetivo.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        regions_cfg_path: str = "configs/regions.yaml",
        rl_cfg_path: str = "configs/rl_config.yaml",
        scenario_name: str = "normal",
        seed: int = 42,
    ):
        super().__init__()
        self.regions_cfg_path = regions_cfg_path
        self.rl_cfg_path = rl_cfg_path
        self.scenario_name = scenario_name
        self.base_seed = seed

        with open(rl_cfg_path, "r", encoding="utf-8") as f:
            self.rl_cfg = yaml.safe_load(f)

        self.weights = self.rl_cfg["reward_weights"]
        self.discretization_cfg = self.rl_cfg.get("discretization", {})

        # Espacio de acciones: 0 (Cloud A), 1 (Cloud B), 2 (Cloud C)
        self.action_space = spaces.Discrete(3)

        # Espacio de observación continuo de 11 dimensiones:
        # [CI_0, CI_1, CI_2, Util_0, Util_1, Util_2, Wait_0, Wait_1, Wait_2, cpu_req_norm, duration_norm]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(11,),
            dtype=np.float32,
        )

        self.simulator: Optional[CloudSimulator] = None
        self.workloads = []
        self.current_idx = 0
        self._initialize_scenario(self.scenario_name, self.base_seed)

    def _initialize_scenario(self, scenario_name: str, seed: int) -> None:
        self.scenario_name = scenario_name
        self.simulator = CloudSimulator(
            config_path=self.regions_cfg_path,
            scenario_name=scenario_name,
            seed=seed,
        )
        gen = WorkloadGenerator(seed=seed)
        num_tasks = self.simulator.scenario_cfg.get("num_tasks", 100)
        arrival_rate = self.simulator.scenario_cfg.get("arrival_rate_per_h", 18.0)
        self.workloads = gen.generate_trace(
            num_tasks=num_tasks,
            arrival_rate_per_h=arrival_rate,
        )
        self.current_idx = 0

    def _get_obs(self) -> np.ndarray:
        if self.current_idx >= len(self.workloads):
            return np.zeros(11, dtype=np.float32)

        w = self.workloads[self.current_idx]
        obs = np.zeros(11, dtype=np.float32)

        # 0..2: Intensidades de carbono normalizadas
        for r in range(3):
            ci = self.simulator.get_carbon_intensity(r, w.arrival_time_h)
            obs[r] = float(np.clip(ci / 1000.0, 0.0, 1.0))

        # 3..5: Utilizaciones de CPU en [0, 1]
        for r in range(3):
            reg = self.simulator.regions[r]
            active_vcpus = sum(cpu for end, cpu in reg.active_allocations if end > w.arrival_time_h)
            obs[3 + r] = float(np.clip(active_vcpus / reg.config.max_vcpu, 0.0, 1.0))

        # 6..8: Tiempos de espera normalizados
        for r in range(3):
            wait = self.simulator.estimate_wait_time(r, w.arrival_time_h, w.cpu_req)
            obs[6 + r] = float(np.clip(wait / 4.0, 0.0, 1.0))

        # 9..10: Requerimientos de la tarea normalizados
        obs[9] = float(w.cpu_req / 16.0)
        obs[10] = float(np.clip(w.duration_h / 4.0, 0.0, 1.0))

        return obs

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        eff_seed = seed if seed is not None else self.base_seed
        scen = self.scenario_name
        if options and "scenario_name" in options:
            scen = options["scenario_name"]

        self._initialize_scenario(scen, eff_seed)
        self.simulator.reset(seed=eff_seed)

        obs = self._get_obs()
        current_w = self.workloads[0] if self.workloads else None
        info = {"workload": current_w, "scenario": self.scenario_name}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        w = self.workloads[self.current_idx]
        act = int(action)

        # Evaluar emisiones y costos potenciales en las 3 regiones para normalización min-max por tarea
        es, cs = [], []
        for r in range(3):
            reg = self.simulator.regions[r]
            wait_r = self.simulator.estimate_wait_time(r, w.arrival_time_h, w.cpu_req)
            t_start_r = w.arrival_time_h + wait_r
            ci_r = self.simulator.get_carbon_intensity(r, t_start_r)
            es.append(w.cpu_req * reg.config.power_kw_per_vcpu * w.duration_h * ci_r)
            cs.append(w.cpu_req * w.duration_h * reg.config.price_per_vcpu_h)

        metric = self.simulator.step_task(w, act)
        self.current_idx += 1

        terminated = self.current_idx >= len(self.workloads)
        truncated = False

        # Componentes de recompensa normalizados en [0, 1]
        norm_carbon = (es[act] - min(es)) / (max(es) - min(es) + 1e-9)
        norm_cost = (cs[act] - min(cs)) / (max(cs) - min(cs) + 1e-9)
        overhead_h = metric.net_latency_h + metric.Wait_time_h
        norm_latency = min(1.0, overhead_h / max(w.slack_h, 1e-6))

        sla_penalty = (
            self.weights["lambda_sla"] * (w.priority / 3.0)
            if metric.sla_violated
            else 0.0
        )

        reward = -(
            self.weights["w_carbon"] * norm_carbon
            + self.weights["w_cost"] * norm_cost
            + self.weights["w_latency"] * norm_latency
        ) - sla_penalty

        mo_reward = np.array(
            [-norm_carbon, -norm_cost, -norm_latency, -sla_penalty],
            dtype=np.float32,
        )

        next_obs = self._get_obs()
        next_w = self.workloads[self.current_idx] if not terminated else None

        step_info = {
            "workload": next_w,
            "metric": metric,
            "mo_reward": mo_reward,
        }
        if terminated:
            step_info["episode_summary"] = self.simulator.summarize_episode()

        return next_obs, float(reward), terminated, truncated, step_info


class DiscretizerWrapper(gym.Wrapper):
    """
    Wrapper para discretizar el espacio de estados continuo en 1,458 estados
    para algoritmos tabulares como Tabular Q-Learning (V1).
    27 (Carbono) * 27 (Carga/Cola) * 2 (Urgencia) = 1,458 estados.
    """

    def __init__(self, env: CarbonCloudEnv):
        super().__init__(env)
        self.num_states = 1458
        self.observation_space = spaces.Discrete(self.num_states)
        self.ci_bins = env.discretization_cfg.get("ci_bins", [150.0, 400.0])
        self.load_thresh = env.discretization_cfg.get("load_threshold_medium", 0.70)
        self.urgency_ratio = env.discretization_cfg.get("urgency_slack_ratio", 0.50)

    def discretize_from_simulator(self, simulator: CloudSimulator, workload: Workload) -> int:
        """Convierte el estado del simulador y la tarea entrante al índice de estado [0..1457]."""
        # 1. Bins de carbono por región (3^3 = 27)
        ci_bins = []
        for r in range(3):
            ci = simulator.get_carbon_intensity(r, workload.arrival_time_h)
            if ci < self.ci_bins[0]:
                ci_bins.append(0)
            elif ci <= self.ci_bins[1]:
                ci_bins.append(1)
            else:
                ci_bins.append(2)
        ci_idx = ci_bins[0] * 9 + ci_bins[1] * 3 + ci_bins[2]

        # 2. Bins de carga/cola por región (3^3 = 27)
        load_bins = []
        for r in range(3):
            reg = simulator.regions[r]
            used = sum(cpu for end, cpu in reg.active_allocations if end > workload.arrival_time_h)
            wait = simulator.estimate_wait_time(r, workload.arrival_time_h, workload.cpu_req)
            u_ratio = used / reg.config.max_vcpu
            if u_ratio >= 1.0 or wait > 0.0:
                load_bins.append(2)
            elif u_ratio >= self.load_thresh:
                load_bins.append(1)
            else:
                load_bins.append(0)
        load_idx = load_bins[0] * 9 + load_bins[1] * 3 + load_bins[2]

        # 3. Bin de urgencia (2 estados)
        slack_ratio = workload.slack_h / max(workload.duration_h, 1e-6)
        urg_idx = 1 if (slack_ratio < self.urgency_ratio or workload.priority == 3) else 0

        # Combinación canónica: CI (27) * 54 + Load (27) * 2 + Urg (2) = 1458
        return int(ci_idx * 54 + load_idx * 2 + urg_idx)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        w = info["workload"]
        discrete_state = (
            self.discretize_from_simulator(self.env.simulator, w)
            if w is not None
            else 0
        )
        return discrete_state, info

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        w = info["workload"]
        if terminated or w is None:
            discrete_state = 0
        else:
            discrete_state = self.discretize_from_simulator(self.env.simulator, w)
        return discrete_state, reward, terminated, truncated, info
