# scripts/train_agent.py
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.env.gym_env import CarbonCloudEnv, DiscretizerWrapper
from src.schedulers.q_learning import TabularQLearningAgent


def train_q_learning():
    rl_cfg_path = "configs/rl_config.yaml"
    with open(rl_cfg_path, "r", encoding="utf-8") as f:
        rl_cfg = yaml.safe_load(f)["q_learning"]

    base_env = CarbonCloudEnv(
        regions_cfg_path="configs/regions.yaml",
        rl_cfg_path=rl_cfg_path,
        scenario_name="normal",
        seed=1000,
    )
    env = DiscretizerWrapper(base_env)

    agent = TabularQLearningAgent(
        num_states=rl_cfg["num_states"],
        num_actions=rl_cfg["num_actions"],
        alpha=rl_cfg["alpha"],
        gamma=rl_cfg["gamma"],
        epsilon_start=rl_cfg["epsilon_start"],
        epsilon_min=rl_cfg["epsilon_min"],
        epsilon_decay=rl_cfg["epsilon_decay"],
        seed=1000,
        training=True,
    )

    scenarios = ["normal", "high_demand", "high_carbon_region"]
    num_episodes = int(rl_cfg["num_episodes"])
    history = []

    print(f"=== INICIANDO ENTRENAMIENTO Q-LEARNING ({num_episodes} EPISODIOS) ===")
    for ep in range(num_episodes):
        scen = scenarios[ep % len(scenarios)] if rl_cfg.get("mixed_scenarios", True) else "normal"
        ep_seed = 1000 + ep  # Semillas disjuntas de la semilla de evaluación (42)

        state, _ = env.reset(seed=ep_seed, options={"scenario_name": scen})
        terminated = False
        total_reward = 0.0

        while not terminated:
            action = agent.select_action(state)
            next_state, reward, terminated, _, info = env.step(action)
            agent.update(state, action, reward, next_state, terminated)
            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        summary = info["episode_summary"]
        history.append({
            "episode": ep + 1,
            "scenario": scen,
            "total_reward": total_reward,
            "epsilon": agent.epsilon,
            "total_co2_g": summary["total_co2_g"],
            "total_cost_usd": summary["total_cost_usd"],
            "sla_violation_rate_pct": summary["sla_violation_rate_pct"],
        })

        if (ep + 1) % 100 == 0:
            recent_r = np.mean([h["total_reward"] for h in history[-60:]])
            visited_pct = np.mean(np.any(agent.q_table != 0.0, axis=1)) * 100.0
            print(
                f"Episodio {ep + 1:4d}/{num_episodes} | Recompensa Media (últ. 60): {recent_r:7.2f} "
                f"| Epsilon: {agent.epsilon:.3f} | Estados visitados: {visited_pct:.1f}%"
            )

    # 1. Guardar política entrenada congelada
    model_path = "models_saved/q_table_v1.npy"
    agent.save(model_path)
    print(f"\n[OK] Q-Table guardada en: {model_path}")

    # 2. Guardar log de entrenamiento y generar curva de convergencia
    os.makedirs("results", exist_ok=True)
    df_hist = pd.DataFrame(history)
    df_hist.to_csv("results/training_log.csv", index=False)

    plt.figure(figsize=(9, 4.5))
    for scen in scenarios:
        sub = df_hist[df_hist["scenario"] == scen]
        rolling = sub["total_reward"].rolling(window=15, min_periods=1).mean()
        plt.plot(sub["episode"], rolling, label=f"Media móvil ({scen})", linewidth=2)

    plt.title("Convergencia del Entrenamiento - Agente Q-Learning Tabular (V1)")
    plt.xlabel("Episodio de Entrenamiento")
    plt.ylabel("Recompensa Acumulada por Episodio")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = "results/training_curve.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[OK] Gráfica de convergencia generada en: {plot_path}")


if __name__ == "__main__":
    train_q_learning()