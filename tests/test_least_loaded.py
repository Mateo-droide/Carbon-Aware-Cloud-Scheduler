# tests/test_least_loaded.py
# from src.env.cloud_simulator import CloudSimulator
# from src.schedulers.least_loaded import LeastLoadedScheduler
# from src.models.schemas import Workload

# def test_least_loaded_tie_breaker():
#     """
#     Si todas las regiones están vacías (espera = 0), debe elegir la más barata.
#     Cloud B cuesta $0.025, Cloud C cuesta $0.040, Cloud A cuesta $0.060.
#     """
#     sim = CloudSimulator(config_path="configs/regions.yaml", scenario_name="normal")
#     scheduler = LeastLoadedScheduler(simulator=sim)
    
#     # Creamos un workload genérico (req: 4 vCPUs, 1 hora)
#     w = Workload(task_id=1, arrival_time_h=0.0, cpu_req=4, duration_h=1.0, priority=2, slack_h=1.0, deadline_h=2.0)
    
#     # Como el clúster acaba de inicializarse, todas las colas están en 0.
#     chosen_region = scheduler.select_action(obs=w)
    
#     # Debe elegir Cloud B (ID = 1) por ser la de menor costo
#     assert chosen_region == 1, f"Error: Eligió la región {chosen_region} en lugar de la 1 (Cloud B)."

# def test_least_loaded_avoids_saturation():
#     """
#     Si la región más barata está saturada, debe elegir una más cara pero libre.
#     """
#     sim = CloudSimulator(config_path="configs/regions.yaml", scenario_name="normal")
#     scheduler = LeastLoadedScheduler(simulator=sim)
    
#     # Saturamos intencionalmente Cloud B (ID 1) y Cloud C (ID 2)
#     max_cpu_b = sim.regions[1].config.max_vcpu
#     max_cpu_c = sim.regions[2].config.max_vcpu
#     sim.regions[1].active_allocations.append((10.0, max_cpu_b)) # Ocupada hasta la hora 10
#     sim.regions[2].active_allocations.append((10.0, max_cpu_c)) # Ocupada hasta la hora 10
    
#     w = Workload(task_id=2, arrival_time_h=0.0, cpu_req=4, duration_h=1.0, priority=2, slack_h=1.0, deadline_h=2.0)
    
#     chosen_region = scheduler.select_action(obs=w)
    
#     # Debe elegir Cloud A (ID = 0) porque B y C tienen tiempo de espera > 0
#     assert chosen_region == 0, f"Error: Eligió la región {chosen_region}, que está saturada, en lugar de la 0."