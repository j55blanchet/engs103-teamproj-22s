"""
    optimization.py

    Finds an optimal set of routes for ships performing a transpacific voyage
    from asian last-port-of-calls to ports on the American west-coast.
"""

import math
import numpy as np
from typing import *
import gurobipy
from gurobipy import Model, GRB


def optimize_transpacific_cargo_routing(
  asian_port_names: List[str] = ["Busan", "Hong Kong", "Singapore"],
  american_port_names: List[str] = ["Seattle", "Los Angeles", "San Francisco"],
  port_distance_matrix: np.ndarray = np.array([
    # Format: |         | Busan | Hong Kong | Singapore | Seattle | Log Angeles | San Fran
    #         | Seattle |  4607 |      5740 |      7062 |       0 |        1148 |      556
    #         | Los A.  |  5529 |      6380 |      7669 |    1148 |           0 |      622
    #         | San Fran|  4919 |      5918 |      7901 |     556 |          622|        0
    [4607.0, 5740.0, 7062.0,    0.0, 1148.0, 556.0],
    [5529.0, 6280.0, 7669.0, 1148.0,    0.0, 622.0],
    [4919.0, 5918.0, 7901.0,  556.0,  622.0,   0.0]
  ]),
  container_demand_matrix: np.ndarray = np.array([
    # Format: |         | Busan | Hong Kong | Singapore
    #         | Seattle |   75 |        75 |        50
    #         | Los A.  |   100 |       180 |        70
    #         | San Fran|   70  |        20 |        40
    [105, 85,  70],
    [140, 190, 90],
    [70,  20,  40]
  ]),
  # Names, used for debugging / printing
  vessel_names: List[str]        = ["ShipA", "ShipB", "ShipC", "ShipD", "ShipE", "ShipF", "ShipG", "ShipH", "ShipI", "ShipJ"],
  # Vessel initial ports (index i of asian port)
  vessel_origins: List[int]      = [     0,       0,       0,       1,       1,       1,       1,       2,       2,       2],
  vessel_capacities: List[int]   = [    60,      80,     105,     110,     110,      70,      80,      70,      50,     110],
  vessel_maxspeeds: List[float]  = [    24,      23,      20,      22,      21,      24,      22,      26,      25,      22],
  vessel_costfactor: List[float] = [  19.5,    22.5,    30.5,    32.0,    31.0,    24.0,    25.0,    23.0,    15.0,    31.0]
) -> gurobipy.Model:
    
    asian_port_count = len(asian_port_names)
    american_port_count = len(american_port_names)
    all_port_names = asian_port_names + american_port_names
    all_port_count = asian_port_count + american_port_count

    m = gurobipy.Model('Transpacific Cargo Routing')

    vessel_variables = []
    for k in range(len(vessel_names)):
      vessel_k_variables = {}
      vessel_variables.append(vessel_k_variables)

      ckj_cargo_unloading_variables = m.addVars(
        [
          f'c_{k}-{j+asian_port_count}'
          for j in range(american_port_count)
        ],
        vtype=GRB.INTEGER,
        lb=0,
        obj=0,
        name=f"Vessel {k} Cargo"
      )
      vessel_k_variables["cargo_unloading"] = ckj_cargo_unloading_variables


      xkij_transpacific_crossing_variables = m.addVars(
        [
          f'x_{k}_{i}_{asian_port_count+j}'
          for i in range(asian_port_count)
          for j in range(american_port_count)
          if vessel_origins[k] == i
        ],
        vtype=GRB.BINARY,
        lb=0,
        ub=1,
        obj=[
          port_distance_matrix[j, i] * vessel_costfactor[k]
          # f'x-crosstranspacific-ves{k}-port{i}-port{j}'
          for i in range(asian_port_count)
          for j in range(american_port_count)
          if vessel_origins[k] == i
        ],
        name=f"Vessel {k} Pacific Crossing"
      )
      vessel_k_variables["transpacific_crossing"] = xkij_transpacific_crossing_variables

      ykij_americancoast_transit_variables = m.addVars(
        [
          f'y_{k}-{j1 + asian_port_count}-{j2 + asian_port_count}'
          for j1 in range(american_port_count)
          for j2 in range(american_port_count)
          if j1 != j2
        ],
        vtype=GRB.BINARY,
        lb=0,
        ub=1,
        obj=[
          # f'x-crosstranspacific-ves{k}-port{i}-port{j}'
          port_distance_matrix[j1, asian_port_count + j2] * vessel_costfactor[k]
          for j1 in range(american_port_count)
          for j2 in range(american_port_count)
          if j1 != j2
        ],
        name=f"Vessel {k} American Port Routes"
      )
      vessel_k_variables["american_routes"] = ykij_americancoast_transit_variables
      

    # Vessel's cargo must not exceed capacity
    m.addConstrs(
      (
          sum([
            vessel_variables[k]["cargo_unloading"][f"c_{k}-{asian_port_count + j}"]
            for j in range(american_port_count)
          ]) <= vessel_capacities[k]
          for k in range(len(vessel_names)
        )
      ),
      name="Vessel Capacity Constraints"
    )

    # Each vessel must cross the ocean only once - from its origin to 
    # some american west coast destination
    m.addConstrs(
      (
        sum(
          vessel_variables[k]["transpacific_crossing"][f"x_{k}_{i}_{asian_port_count + j}"]
          for i in range(asian_port_count)
          for j in range(american_port_count)
          if vessel_origins[k] == i
        ) == 1
        for k in range(len(vessel_names))
      ),
      name="Transpacific Crossing Constraints"
    )

    # TODO:
    # Each vessel can only leave an american port it has arrived at. Furthermore, it 
    # can only arrive at each port at most once. 
    # m.addConstrs(
    #   (
    #     k for k in range(len(vessel_names))
    #   ),
    #   name="Route integrity constraints"
    # )

    # TODO: Cargo demand must be met

    return m, vessel_variables





if __name__ == "__main__":
  model, vessel_variables = optimize_transpacific_cargo_routing()
  print(model, vessel_variables)
  print('Done')