
# Informe Técnico - Primer Parcial (V1): Carbon-Aware Cloud Scheduler

## 1. Planteamiento del Problema y Estado del Arte Inicial
El despliegue de cargas de trabajo en infraestructuras de nube geodistribuidas enfrenta un conflicto estructural entre el rendimiento operativo (latencia y cumplimiento de acuerdos de nivel de servicio, SLA), el costo monetario de cómputo y la sostenibilidad ambiental medida a través de las emisiones operativas de dióxido de carbono ($\text{CO}_2$). Los planificadores tradicionales como *Round Robin* asignan recursos de forma cíclica ignorando tanto la saturación heterogénea de los nodos como la intensidad de carbono de la red eléctrica local. A su vez, las heurísticas orientadas exclusivamente a rendimiento o costo (como *Least Loaded* con desempate económico) tienden a concentrar la carga en centros de datos alimentados por combustibles fósiles debido a que la energía eléctrica más barata suele presentar mayor huella de carbono (Zhao et al., CFWS; Sukprasert et al., EuroSys '24).

La literatura científica reciente demuestra que la intensidad de carbono varía entre $6\times$ y $8\times$ entre regiones geográficas (desde $11\text{--}45\text{ gCO}_2\text{eq/kWh}$ en redes con predominio hidroeléctrico, eólico o nuclear hasta $490\text{--}820\text{ gCO}_2\text{eq/kWh}$ en redes dependientes de gas natural y carbón) y fluctúa horariamente según la disponibilidad solar y eólica (Meta Carbon Explorer; Souza et al., CASPER, IGSC '23). Obras como *CEDTS-RL*, *Google CICS* (Radovanović et al., 2023) y *MO-PPO* (Chen & Chen, 2025) formulan la asignación espacial de tareas como un Proceso de Decisión de Markov (MDP) resuelto mediante Aprendizaje por Refuerzo (RL), logrando reducciones sustanciales de emisiones sin degradar los tiempos de respuesta.

*(Nota: La matriz comparativa detallada de los 7 artículos científicos verificados se encuentra en `docs/literature_matrix.md`).*

---

## 2. Objetivo, Pregunta de Investigación e Hipótesis

### Objetivo General (Versión 1)
Diseñar, implementar y evaluar experimentalmente un simulador modular de planificación en la nube con tres regiones heterogéneas (Cloud A, B y C) y colas de espera FIFO, comparando un planificador estático (*Round Robin*), un planificador heurístico consciente de carga y costo (*Least Loaded*) y un agente entrenado mediante Aprendizaje por Refuerzo (*Tabular Q-Learning*) bajo escenarios de operación normal, alta demanda y alta intensidad de carbono regional.

### Pregunta de Investigación
¿En qué medida un planificador basado en Aprendizaje por Refuerzo (*Q-Learning* tabular), entrenado con una señal de recompensa compuesta por emisiones de $\text{CO}_2$, costo operativo, latencia de cola/red y penalización por violación de SLA, logra reducir la huella de carbono acumulada sin comprometer la estabilidad de las colas ni incrementar significativamente los costos frente a políticas estáticas (*Round Robin*) y heurísticas voraces (*Least Loaded*)?

### Hipótesis Medibles
* **Hipótesis 1 (Compromiso Ambiental en Operación Normal y Alto Carbono):** Bajo condiciones normales y de pico de carbono regional, el agente *Q-Learning* reducirá las emisiones totales de $\text{CO}_2$ respecto a *Round Robin* y *Least Loaded* al desviar selectivamente cargas hacia la región de energía limpia (Cloud A) y moderada (Cloud B), asumiendo un incremento acotado en el costo frente a la heurística más barata (*Least Loaded*).
* **Hipótesis 2 (Protección de SLA en Alta Demanda):** En el escenario de alta demanda ($34\text{ tareas/h}$), donde la capacidad de la región más limpia (Cloud A, $64\text{ vCPUs}$) resulta insuficiente, el agente *Q-Learning* reducirá drásticamente la tasa de violaciones de SLA frente a *Round Robin* al incorporar el estado de congestión de las colas FIFO en su espacio de observación.
* **Hipótesis 3 (Límite de Resolución de la Discretización Fija - Puente a V2):** Un agente tabular con umbrales de discretización estáticos y pesos fijos exhibirá rigidez cuando dos niveles extremos de carbono superen el mismo umbral superior de cuantización, justificando la transición hacia espacios de estado continuos (*PPO / MO-PPO*), pesos dinámicos y análisis del Frente de Pareto en el Segundo Parcial (V2).

---

## 3. Modelo Matemático y Diseño Experimental

### 3.1 Parametrización de las Tres Regiones Cloud
| Región | Perfil Operativo | Intensidad Media $\mu_{CI}$ ($\text{gCO}_2/\text{kWh}$) | Costo $p_r$ ($\text{USD/vCPU-h}$) | Capacidad $C_{r,\max}$ ($\text{vCPUs}$) | Latencia Red $L_{r,\text{net}}$ ($\text{h}$) | Potencia $P_{\text{vCPU},r}$ ($\text{kW/vCPU}$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Cloud A (0)** | Verde / Cara / Lejana | $45 \pm 15$ | $\$0.060$ | $64$ | $0.120$ | $0.035$ |
| **Cloud B (1)** | Económica / Carbono Medio | $250 \pm 90$ | $\$0.025$ | $96$ | $0.060$ | $0.040$ |
| **Cloud C (2)** | Rápida / Alta Cap. / Fósil | $550 \pm 120$ ($780$ en Esc. 3) | $\$0.040$ | $160$ | $0.015$ | $0.045$ |

### 3.2 Ecuaciones Físicas y Formulación del MDP
Para cada trabajo $w_i = (t_i, c_i, d_i, \text{prio}_i, DDL_i)$ asignado a la región $r \in \{0, 1, 2\}$:
1. **Energía y Emisiones:** $\text{Energía}_i = c_i \cdot P_{\text{vCPU},r} \cdot d_i\text{ (kWh)}$; $\quad E_i = \text{Energía}_i \cdot CI_r(t_{i,\text{start}})\text{ (gCO}_2\text{)}$.
2. **Costo Operativo:** $\text{Cost}_i = c_i \cdot d_i \cdot p_r\text{ (USD)}$.
3. **Latencia Real y Cola FIFO:** $D_{i,\text{real}} = d_i + L_{r,\text{net}} + D_{i,\text{wait}}$; ocurre violación de SLA ($v_i = 1$) si $t_i + D_{i,\text{real}} > DDL_i$.
4. **Espacio de Estados Discretizado ($1,458$ estados):** Producto cartesiano del nivel de carbono $b(CI_r) \in \{0,1,2\}^3$ ($27$), estado de ocupación/cola FIFO $b(\text{Load}_r) \in \{0,1,2\}^3$ ($27$) y urgencia de la tarea $b(\text{Urg}_i) \in \{0,1\}$ ($2$).
5. **Recompensa Normalizada:** $r_t = -(0.50\hat{E}_i + 0.25\widehat{\text{Cost}}_i + 0.25\hat{L}_i) - 2.0 \cdot v_i \cdot (\text{prio}_i / 3)$.

---

## 4. Resultados Experimentales (Evaluación sobre 10 Semillas Disjuntas: 200..209)

El agente *Q-Learning* fue entrenado durante $600$ episodios (semillas `1000..1599`) y evaluado con su tabla $Q$ congelada ($\epsilon = 0$) frente a *Round Robin* y *Least Loaded* en $10$ semillas de prueba independientes (`200..209`, $100$ tareas por semilla):

| Escenario | Scheduler | Emisiones $\text{CO}_2$ ($\text{g}$) | Costo Total ($\text{USD}$) | Latencia Overhead ($\text{h}$) | Violaciones SLA ($\%$) | Uso Cloud A / B / C ($\%$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Normal** ($18\text{ tareas/h}$) | Round Robin | $10,556.3 \pm 1,346.6$ | $\$43.80 \pm 6.11$ | $0.1606$ | $5.0\% \pm 5.4\%$ | $34.0 / 33.0 / 33.0$ |
| | Least Loaded | $14,680.9 \pm 2,790.6$ | $\mathbf{\$34.38 \pm 5.98}$ | $\mathbf{0.0448}$ | $\mathbf{0.0\% \pm 0.0\%}$ | $0.7 / 64.6 / 34.7$ |
| | **Q-Learning (V1)** | $\mathbf{9,324.1 \pm 2,485.5}$ | $\$39.47 \pm 6.51$ | $0.0882$ | $0.1\% \pm 0.3\%$ | $\mathbf{21.6 / 66.5 / 11.9}$ |
| **High Demand** ($34\text{ tareas/h}$) | Round Robin | $\mathbf{10,705.2 \pm 1,418.7}$ | $\$43.80 \pm 6.11$ | $0.4241$ | $16.4\% \pm 7.5\%$ | $34.0 / 33.0 / 33.0$ |
| | Least Loaded | $14,983.7 \pm 1,652.1$ | $\mathbf{\$39.86 \pm 6.59}$ | $\mathbf{0.0767}$ | $\mathbf{0.6\% \pm 1.9\%}$ | $10.7 / 45.1 / 44.2$ |
| | **Q-Learning (V1)** | $12,429.3 \pm 2,241.1$ | $\$43.09 \pm 6.97$ | $0.1468$ | $3.9\% \pm 5.3\%$ | $\mathbf{22.2 / 45.3 / 32.5}$ |
| **High Carbon** ($18\text{ tareas/h}$) | Round Robin | $13,880.4 \pm 1,790.5$ | $\$43.80 \pm 6.11$ | $0.1606$ | $5.0\% \pm 5.4\%$ | $34.0 / 33.0 / 33.0$ |
| | Least Loaded | $19,765.4 \pm 4,154.5$ | $\mathbf{\$34.38 \pm 5.98}$ | $\mathbf{0.0448}$ | $\mathbf{0.0\% \pm 0.0\%}$ | $0.7 / 64.6 / 34.7$ |
| | **Q-Learning (V1)** | $\mathbf{11,560.3 \pm 3,771.0}$ | $\$39.47 \pm 6.51$ | $0.0882$ | $0.1\% \pm 0.3\%$ | $\mathbf{21.6 / 66.5 / 11.9}$ |

### Gráficas Generadas
* Comparativa de las 4 métricas: `../results/metrics_comparison_v1.png`
* Distribución porcentual por región: `../results/region_distribution_v1.png`
* Curva de convergencia de entrenamiento: `../results/training_curve.png`

---

## 5. Discusión Crítica y Validación de Hipótesis

1. **Confirmación de la Hipótesis 1 (Escenarios Normal y Alto Carbono):**
   * En el escenario **Normal**, *Q-Learning* logra la menor huella de carbono ($9,324.1\text{ gCO}_2$), reduciendo las emisiones un **$-11.7\%$ respecto a Round Robin** y un **$-36.5\%$ respecto a Least Loaded**. Además, reduce las violaciones de SLA de *Round Robin* del $5.0\%$ a apenas $0.1\%$ y ahorra un $9.9\%$ en costo ($\$39.47$ vs. $\$43.80$).
   * En el escenario **Región con Alto Carbono**, la ventaja ambiental de *Q-Learning* se amplifica: reduce las emisiones un **$-16.7\%$ frente a Round Robin** ($11,560.3\text{ g}$ vs. $13,880.4\text{ g}$) y un **$-41.5\%$ frente a Least Loaded** ($19,765.4\text{ g}$). Esto ocurre porque *Least Loaded*, al desempatar por precio cuando las colas están vacías, evita Cloud A ($0.7\%$) y envía el $34.7\%$ de las cargas a Cloud C, mientras que *Q-Learning* restringe el uso de Cloud C al $11.9\%$.
2. **Confirmación de la Hipótesis 2 (Escenario de Alta Demanda):**
   * Cuando la tasa de llegada sube a $34\text{ tareas/h}$, *Round Robin* satura la cola FIFO de Cloud A ($64\text{ vCPUs}$), elevando la espera media a $0.4241\text{ h}$ y rompiendo el SLA en el **$16.4\%$** de los trabajos.
   * *Q-Learning* detecta el estado `Encolando (bin=2)` en Cloud A y redirige dinámicamente el tráfico excedente hacia Cloud C (subiendo su asignación del $11.9\%$ al $32.5\%$), lo que reduce las violaciones de SLA más de cuatro veces (**del $16.4\%$ al $3.9\%$**) y mantiene un **$-17.0\%$ menos de emisiones de $\text{CO}_2$ que Least Loaded** ($12,429.3\text{ g}$ vs. $14,983.7\text{ g}$).
3. **Confirmación de la Hipótesis 3 y Trabajo Futuro hacia el Segundo Parcial (V2):**
   * Se observa que el reparto porcentual de regiones de *Q-Learning* entre el escenario `normal` y `high_carbon_region` permanece en $21.6\% / 66.5\% / 11.9\%$. La causa matemática exacta es la cuantización discreta: como el umbral alto de carbono se fijó en $400\text{ gCO}_2/\text{kWh}$, tanto la media normal de Cloud C ($550\text{ gCO}_2/\text{kWh}$) como su pico extremo ($780\text{ gCO}_2/\text{kWh}$) se mapean al mismo índice discreto $b(CI_C) = 2$. Esta evidencia empírica justifica directamente la arquitectura del Segundo Parcial (V2), donde conectaremos **PPO / MO-PPO** directamente al vector de observación continuo de 11 dimensiones de `CarbonCloudEnv` e implementaremos **pesos dinámicos y el Frente de Pareto** utilizando el vector `info["mo_reward"]`.
