# Diagramas de Arquitectura y Ciclo de Decisión (Versión 1 - Primer Parcial)

## 1. Arquitectura Modular y Evolutiva (V1 -> V2)

```mermaid
flowchart TD
    subgraph CFG["Capa de Configuración (YAML)"]
        C1["configs/regions.yaml<br/>(Cloud A, B, C + 3 Escenarios)"]
        C2["configs/rl_config.yaml<br/>(Pesos w_c, w_cost, w_lat, lambda_sla)"]
    end

    subgraph CORE["Núcleo de Simulación Física (src/models & src/env)"]
        WG["WorkloadGenerator<br/>(Poisson + Semilla Controlada)"]
        SIM["CloudSimulator<br/>(Colas FIFO por Región + Cálculo Físico)"]
        GYM["CarbonCloudEnv (Gymnasium API)<br/>Estado Continuo [0,1]^11 + Recompensa Escalar/Vectorial"]
        DISC["DiscretizerWrapper (V1)<br/>Cuantización Posicional: 27x27x2 = 1,458 Estados"]
    end

    subgraph SCHED["Módulo Polimórfico de Schedulers (src/schedulers)"]
        BASE["BaseScheduler<br/>select_action(obs, info)"]
        RR["RoundRobinScheduler<br/>(Baseline Cíclico Estático)"]
        LL["LeastLoadedScheduler<br/>(Heurístico Cola FIFO + Menor Costo)"]
        QL["TabularQLearningAgent (V1)<br/>(Tabla Q 1458x3 en NumPy)"]
        V2["[Reserva 2do Parcial - V2]<br/>PPO / MO-PPO + Pesos Dinámicos + Frente de Pareto"]
    end

    subgraph EVAL["Evaluación, Trazas y Visualización (src/evaluation)"]
        RUN["ExperimentRunner<br/>(10 Semillas Disjuntas: 200..209)"]
        OUT1["data/generated_workloads/*.csv"]
        OUT2["results/eval_summary_table.csv"]
        OUT3["results/metrics_comparison_v1.png<br/>results/region_distribution_v1.png"]
    end

    C1 --> WG & SIM
    C2 --> GYM & DISC
    WG --> SIM
    SIM <--> GYM
    GYM --> DISC
    BASE --> RR & LL & QL & V2
    DISC <--> QL
    GYM -. "Estado Continuo + info['mo_reward']" .-> V2
    RR & LL & QL --> RUN
    RUN --> SIM
    RUN --> OUT1 & OUT2 & OUT3