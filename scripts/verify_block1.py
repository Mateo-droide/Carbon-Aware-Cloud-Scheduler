# scripts/verify_block1.py
import yaml
from src.models.workload import WorkloadGenerator
from src.env.cloud_simulator import CloudSimulator
from src.env.gym_env import CarbonCloudEnv, DiscretizerWrapper
from src.schedulers.round_robin import RoundRobinScheduler
from src.schedulers.least_loaded import LeastLoadedScheduler
from src.schedulers.q_learning import TabularQLearningAgent


def run_verification():
    config_path = "configs/regions.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Instanciar el wrapper discretizador para que Q-Learning traduzca obs=w
    dummy_env = CarbonCloudEnv(regions_cfg_path=config_path, rl_cfg_path="configs/rl_config.yaml")
    discretizer = DiscretizerWrapper(dummy_env)

    print("=== COMPARATIVA EXPERIMENTAL V1: ROUND ROBIN vs. LEAST LOADED vs. Q-LEARNING ===")
    for scen_name, scen_params in cfg["scenarios"].items():
        gen = WorkloadGenerator(seed=42)
        workloads = gen.generate_trace(
            num_tasks=scen_params["num_tasks"],
            arrival_rate_per_h=scen_params["arrival_rate_per_h"],
        )

        sim = CloudSimulator(config_path=config_path, scenario_name=scen_name, seed=42)

        q_agent = TabularQLearningAgent(
            training=False,
            simulator=sim,
            discretizer=discretizer,
        )
        q_agent.load("models_saved/q_table_v1.npy")

        schedulers = {
            "Round_Robin": RoundRobinScheduler(num_regions=3),
            "Least_Loaded": LeastLoadedScheduler(sim),
            "Q_Learning_V1": q_agent,
        }

        print(f"\n======================================================================")
        print(f"ESCENARIO: [{scen_name.upper()}] (Tasa llegada: {scen_params['arrival_rate_per_h']} tareas/h)")
        print(f"======================================================================")

        for sched_name, scheduler in schedulers.items():
            sim.reset(seed=42)
            scheduler.reset()

            for w in workloads:
                region_id = scheduler.select_action(obs=w)
                sim.step_task(w, region_id)

            summary = sim.summarize_episode()
            print(
                f" -> {sched_name:14s} | CO2: {summary['total_co2_g']:9.2f} g "
                f"| Costo: ${summary['total_cost_usd']:6.3f} "
                f"| Lat. Overhead: {summary['mean_latency_overhead_h']:.4f} h "
                f"| SLA Viol.: {summary['sla_violation_rate_pct']:5.1f}%"
            )


if __name__ == "__main__":
    run_verification()