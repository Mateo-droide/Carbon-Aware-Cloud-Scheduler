import os
import pandas as pd


def test_generated_artifacts_exist_and_valid():
    """Verifica que los entregables R3, R7 y R8 hayan sido generados sin valores nulos."""
    required_files = [
        "models_saved/q_table_v1.npy",
        "data/generated_workloads/workloads_normal_seed200.csv",
        "results/eval_runs_detailed.csv",
        "results/eval_summary_table.csv",
        "results/metrics_comparison_v1.png",
        "results/region_distribution_v1.png",
    ]
    for filepath in required_files:
        assert os.path.exists(filepath), f"Falta el entregable obligatorio: {filepath}"

    df_summary = pd.read_csv("results/eval_summary_table.csv")
    assert len(df_summary) == 9
    assert (df_summary["total_co2_g_mean"] > 0).all()
    assert (df_summary["total_cost_usd_mean"] > 0).all()
