# tests/test_rl_env_and_agent.py
import numpy as np
from src.env.gym_env import CarbonCloudEnv, DiscretizerWrapper
from src.schedulers.q_learning import TabularQLearningAgent


def test_discretizer_state_bounds():
    base_env = CarbonCloudEnv(seed=42)
    env = DiscretizerWrapper(base_env)
    state, info = env.reset(seed=42)
    assert 0 <= state < 1458
    assert "workload" in info

    next_state, reward, terminated, _, step_info = env.step(0)
    assert 0 <= next_state < 1458
    assert isinstance(reward, float)
    assert step_info["mo_reward"].shape == (4,)


def test_q_learning_bellman_update_and_freeze():
    agent = TabularQLearningAgent(training=True, alpha=0.5, gamma=0.9)
    assert agent.q_table[10, 1] == 0.0

    # En modo entrenamiento debe actualizar la celda Q(10, 1)
    agent.update(state=10, action=1, reward=-0.8, next_state=20, terminated=False)
    assert np.isclose(agent.q_table[10, 1], -0.4)

    # En modo evaluación NO debe modificar la Q-Table
    agent.set_eval_mode()
    agent.update(state=10, action=1, reward=-10.0, next_state=20, terminated=False)
    assert np.isclose(agent.q_table[10, 1], -0.4)