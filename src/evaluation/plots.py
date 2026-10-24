# src/evaluation/plots.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_all_plots(df_raw: pd.DataFrame, out_dir: str = "results") -> None:
    """Genera las gráficas comparativas oficiales del Primer Parcial (Entregable R8)."""
    os.makedirs(out_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    scenario_labels = {
        "normal": "Normal\n(18 tareas/h)",
        "high_demand": "Alta Demanda\n(34 tareas/h)",
        "high_carbon_region": "Región Alto Carbono\n(Cloud C = 780 g/kWh)",
    }
    df_plot = df_raw.copy()
    df_plot["Escenario"] = df_plot["scenario"].map(scenario_labels)

    # 1. Panel 2x2 de las 4 Métricas Obligatorias
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    palette = {"Round Robin": "#7f8c8d", "Least Loaded": "#2980b9", "Q-Learning (V1)": "#27ae60"}

    metrics_spec = [
        ("total_co2_g", "Emisiones Estimadas de CO2 (gCO2)", axes[0, 0]),
        ("total_cost_usd", "Costo Operativo Total (USD $)", axes[0, 1]),
        ("mean_latency_overhead_h", "Latencia Media Red + Cola FIFO (Horas)", axes[1, 0]),
        ("sla_violation_rate_pct", "Violaciones de SLA (%)", axes[1, 1]),
    ]

    for col, title, ax in metrics_spec:
        sns.barplot(
            data=df_plot,
            x="Escenario",
            y=col,
            hue="scheduler",
            palette=palette,
            errorbar="sd",
            capsize=0.08,
            ax=ax,
        )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.legend(title="Scheduler", fontsize=9)

    plt.suptitle(
        "Comparativa Experimental V1: Round Robin vs. Least Loaded vs. Q-Learning (10 Semillas)",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    fig.savefig(f"{out_dir}/metrics_comparison_v1.png", dpi=180)
    plt.close(fig)

    # 2. Distribución de Asignación de Regiones (Cloud A, B, C)
    df_dist = (
        df_raw.groupby(["scenario", "scheduler"], sort=False)[
            ["pct_cloud_A", "pct_cloud_B", "pct_cloud_C"]
        ]
        .mean()
        .reset_index()
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    scenarios_order = ["normal", "high_demand", "high_carbon_region"]

    for idx, scen in enumerate(scenarios_order):
        sub = df_dist[df_dist["scenario"] == scen].set_index("scheduler")
        ax = axes[idx]
        bottom = np.zeros(len(sub))

        colors_reg = ["#2ecc71", "#3498db", "#e74c3c"]
        labels_reg = ["Cloud A (Verde)", "Cloud B (Barata)", "Cloud C (Rápida/Fósil)"]
        cols_reg = ["pct_cloud_A", "pct_cloud_B", "pct_cloud_C"]

        for c_col, label, color in zip(cols_reg, labels_reg, colors_reg):
            values = sub[c_col].values
            ax.bar(sub.index, values, bottom=bottom, label=label, color=color, width=0.55)
            bottom += values

        ax.set_title(scenario_labels[scen].replace("\n", " "), fontsize=10, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.tick_params(axis="x", rotation=15)
        if idx == 0:
            ax.set_ylabel("Porcentaje de Tareas Asignadas (%)")
        if idx == 2:
            ax.legend(loc="upper right", fontsize=8.5)

    plt.suptitle(
        "Política de Enrutamiento por Región (% de Workloads enviados a Cloud A, B y C)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(f"{out_dir}/region_distribution_v1.png", dpi=180, bbox_inches="tight")
    plt.close(fig)