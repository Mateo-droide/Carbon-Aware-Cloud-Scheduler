# scripts/run_experiments.py
from src.evaluation.runner import ExperimentRunner
from src.evaluation.plots import generate_all_plots


def main():
    print("=== EJECUTANDO BATERÍA EXPERIMENTAL OFICIAL V1 (10 SEMILLAS: 200..209) ===")
    runner = ExperimentRunner()
    df_raw, df_summary = runner.run_all()

    generate_all_plots(df_raw, out_dir="results")

    print("\n=== TABLA RESUMEN DE RESULTADOS (MEDIA ± DESVIACIÓN ESTÁNDAR) ===")
    for _, row in df_summary.iterrows():
        print(
            f"[{row['scenario']:18s}] {row['scheduler']:16s} | "
            f"CO2: {row['total_co2_g_mean']:8.1f} ± {row['total_co2_g_std']:5.1f} g | "
            f"Costo: ${row['total_cost_usd_mean']:6.2f} ± {row['total_cost_usd_std']:4.2f} | "
            f"Lat.Overhead: {row['mean_latency_overhead_h_mean']:.4f} h | "
            f"SLA Viol: {row['sla_violation_rate_pct_mean']:4.1f}% ± {row['sla_violation_rate_pct_std']:3.1f}% | "
            f"Regiones(A/B/C): {row['pct_cloud_A_mean']:4.1f}% / {row['pct_cloud_B_mean']:4.1f}% / {row['pct_cloud_C_mean']:4.1f}%"
        )

    print("\n[OK] Artefactos generados correctamente:")
    print("  - Dataset de trazas (R3) : data/generated_workloads/workloads_*_seed200.csv")
    print("  - Métricas por corrida   : results/eval_runs_detailed.csv")
    print("  - Tabla estadística V1   : results/eval_summary_table.csv")
    print("  - Gráfica 4 métricas (R8): results/metrics_comparison_v1.png")
    print("  - Gráfica uso regiones   : results/region_distribution_v1.png")


if __name__ == "__main__":
    main()