# Carbon-Aware Cloud Scheduler (V1 - Primer Parcial)

Sistema de planificación inteligente de cargas de trabajo en la nube sensible a la intensidad de carbono, costo operativo, capacidad de cómputo (colas FIFO) y acuerdos de nivel de servicio (SLA). Diseñado con arquitectura modular compatible con **Gymnasium** para evolucionar hacia optimización multiobjetivo avanzada en el Segundo Parcial (V2).

## Estructura del Proyecto
- `configs/`: Parametrización de Cloud A, B y C, escenarios experimentales (`regions.yaml`) e hiperparámetros RL (`rl_config.yaml`).
- `src/models/`: Contratos de datos tipados (`schemas.py`) y generador estocástico reproducible (`workload.py`).
- `src/env/`: Simulador físico con colas FIFO (`cloud_simulator.py`) y entorno `CarbonCloudEnv` + `DiscretizerWrapper` (`gym_env.py`).
- `src/schedulers/`: Planificadores `RoundRobinScheduler`, `LeastLoadedScheduler` y `TabularQLearningAgent`.
- `src/evaluation/`: Ejecutor estadístico multi-semilla (`runner.py`) y generador de gráficas (`plots.py`).
- `scripts/`: Scripts ejecutables de entrenamiento (`train_agent.py`), verificación rápida (`verify_block1.py`) y batería oficial (`run_experiments.py`).
- `docs/`: Matriz bibliográfica (`literature_matrix.md`), diagramas (`architecture_diagrams.md`) e informe técnico (`report_v1.md`).
- `tests/`: Suite de pruebas unitarias automatizadas en `pytest`.

---

## Instrucciones de Ejecución Reproducible

### Opción 1: Ejecución Local (Nivel 1)
```bash
# 1. Crear y activar entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar suite de pruebas unitarias (pytest)
pytest

# 4. Entrenar el agente Q-Learning Tabular (600 episodios, semillas 1000..1599)
python -m scripts.train_agent

# 5. Ejecutar evaluación oficial V1 (10 semillas de prueba disjuntas: 200..209)
python -m scripts.run_experiments


##Nivel 2: Ejecución Reproducible con Docker (Contenerizada)
Empaqueta el entorno con python:3.12-slim y sus dependencias exactas para garantizar que cualquier evaluador obtenga los mismos resultados con un solo comando sin configurar Python localmente:
```bash

docker compose up --build
