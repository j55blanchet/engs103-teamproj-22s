"""
    optimization.py

    Finds an optimal set of routes for ships performing a transpacific voyage
    from asian last-port-of-calls to ports on the American west-coast.
"""

import math
import numpy as np
from typing import *
import gurobipy


def optimize_transpacific_cargo_routing(
  asian_port_names: List[str] = ["Busan", "Hong Kong", "Singapore"],
  american_port_names: List[str] = ["Seattle", "Los Angeles"],
  port_distance_matrix: np.ndarray = np.array([
    # Format: |         | Busan | Hong Kong | Singapore | Seattle | Log Angeles
    #         | Seattle |  4607 |      5740 |      7062 |       0 |        1148
    #         | Los A.  |  5529 |      6380 |      7669 |    1148 |           0
    [4607.0, 5740.0, 7062.0,    0.0, 1148.0],
    [5529.0, 6280.0, 7669.0, 1148.0,    0.0]
  ]),
  container_demand_matrix: np.ndarray = np.ndarray([
    # Format: |         | Busan | Hong Kong | Singapore
    #         | Seattle |   105 |        85 |        70
    #         | Los A.  |   140 |       190 |        90
    [105, 85,  70],
    [140, 190, 90]
  ]),
  # Names, used for debugging / printing
  vessel_names: List[str]        = ["ShipA", "ShipB", "ShipC", "ShipD", "ShipE", "ShipF", "ShipG", "ShipH", "ShipI", "ShipJ"],
  # Vessel initial ports (index i of asian port)
  vessel_origins: List[int]      = [     0,       0,       0,       1,       1,       1,       1,       2,       2,       2],
  vessel_capacities: List[int]   = [    60,      80,     105,     110,     110,      70,      80,      70,      50,     110],
  vessel_maxspeeds: List[float]  = [    24,      23,      20,      22,      21,      24,      22,      26,      25,      22],
  vessel_costfactor: List[float] = [  19.5,    22.5,    30.5,    32.0,    31.0,    24.0,    25.0,    23.0,    15.0,    31.0]
) -> gurobipy.Model:
    
    m = gurobipy.Model('Transpacific Cargo Routing')

    return m