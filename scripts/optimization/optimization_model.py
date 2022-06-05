"""
    optimization.py

    Finds an optimal set of routes for ships performing a transpacific voyage
    from asian last-port-of-calls to ports on the American west-coast.
"""

from dataclasses import dataclass
import math
import numpy as np
from typing import *
import gurobipy
from gurobipy import Model, GRB

@dataclass
class OutputVesselRouteRow:
  ship_name: str
  capacity_teu: int
  max_speed: float
  est_cost_per_nm_at_18knots: float
  start_port: str
  port_1: Union[str, None]
  cargo_1: Union[int, None]
  dist_1: Union[float, None]
  port_2: Union[str, None]
  cargo_2: Union[int, None]
  dist_2: Union[float, None]
  port_3: Union[str, None]
  cargo_3: Union[int, None]
  dist_3: Union[float, None]
  port_4: Union[str, None]
  cargo_4: Union[int, None]
  dist_4: Union[float, None]
  port_5: Union[str, None]
  cargo_5: Union[int, None]
  dist_5: Union[float, None]

  def csv_row(self) -> list:
    return [
      self.ship_name,
      self.capacity_teu,
      self.max_speed,
      self.est_cost_per_nm_at_18knots,
      self.start_port,
      self.port_1,
      self.cargo_1,
      self.dist_1,
      self.port_2,
      self.cargo_2,
      self.dist_2,
      self.port_3,
      self.cargo_3,
      self.dist_3,
      self.port_4,
      self.cargo_4,
      self.dist_4,
      self.port_5,
      self.cargo_5,
      self.dist_5,
    ]
  
  @staticmethod
  def csv_header_row() -> list:
    return [
      "ship_name",
      "capacity_teu",
      "max_speed",
      "est_cost_per_nm_at_18knots",
      "start_port",
      "port_1",
      "cargo_1",
      "dist_1",
      "port_2",
      "cargo_2",
      "dist_2",
      "port_3",
      "cargo_3",
      "dist_3",
      "port_4",
      "cargo_4",
      "dist_4",
      "port_5",
      "cargo_5",
      "dist_5",
    ]



class TranspacificCargoRoutingProblem():

  def __init__(self,
    asian_port_names: List[str] = ["Busan", "Hong Kong", "Singapore"],
    american_port_names: List[str] = ["Seattle", "Los Angeles", "San Francisco"],
    port_distance_matrix: np.ndarray = np.array([
      # Format: |         | Busan | Hong Kong | Singapore | Seattle | Log Angeles | San Fran
      #         | Seattle |  4607 |      5740 |      7062 |       0 |        1148 |      656
      #         | Los A.  |  5529 |      6380 |      7669 |    1148 |           0 |      622
      #         | San Fran|  4919 |      5918 |      7901 |     656 |          622|        0
      [4607.0, 5740.0, 7062.0,    0.0, 1148.0, 656.0],
      [5529.0, 6280.0, 7669.0, 1148.0,    0.0, 622.0],
      [4919.0, 5918.0, 7901.0,  656.0,  622.0,   0.0]
    ]),
    port_demand_matrix: np.ndarray = np.array([
      # Format: |         | Busan | Hong Kong | Singapore
      #         | Seattle |   75 |        75 |         50
      #         | Los A.  |   100 |       180 |        70
      #         | San Fran|   70  |        20 |        40
      [75,  75,  50],
      [100, 180, 70],
      [70,  20,  40]
    ]),
    # Names, used for debugging / printing
    vessel_names: List[str]        = ["ShipA", "ShipB", "ShipC", "ShipD", "ShipE", "ShipF", "ShipG", "ShipH", "ShipI", "ShipJ"],
    # Vessel initial ports (index i of asian port)
    vessel_origins: List[int]      = [     0,       0,       0,       1,       1,       1,       1,       2,       2,       2],
    vessel_capacities: List[int]   = [    60,      80,     115,     110,     110,      70,      80,      70,      50,     110],
    vessel_maxspeeds: List[float]  = [    24,      23,      20,      22,      21,      24,      22,      26,      25,      22],
    vessel_costfactor: List[float] = [  19.5,    22.5,    30.5,    32.0,    31.0,    24.0,    25.0,    23.0,    15.0,    31.0],
    enforce_integer_cargo: bool = True,
    add_duplicate_constraints: bool = True,
    enable_gurobi_logging: bool = True,
  ) -> None:

    self.asian_port_names = asian_port_names
    self.longest_asian_port_name = max(len(p) for p in self.asian_port_names)
    self.american_port_names = american_port_names
    self.longest_american_port_name = max(len(p) for p in self.american_port_names)
    self.port_distance_matrix = port_distance_matrix
    self.port_demand_matrix = port_demand_matrix
    self.vessel_names = vessel_names
    self.longest_vessel_name = max(len(v) for v in vessel_names)
    self.vessel_origins = vessel_origins
    self.vessel_capacities = vessel_capacities
    self.vessel_maxspeeds = vessel_maxspeeds
    self.vessel_costfactor = vessel_costfactor
    self.enforce_integer_cargo = enforce_integer_cargo
    self.add_duplicate_constraints = add_duplicate_constraints
    self.enable_gurobi_logging = enable_gurobi_logging

    self.m = gurobipy.Model('Transpacific Cargo Routing Problem')

    self.vessel_variables = [
      self._create_vessel_variables(k)
      for k in range(self.vessel_count)
    ]

    self._add_constraint_cargo_demand_must_be_satisfied()
    self._add_transpacific_singlecrossing_constraints()

    for k in range(self.vessel_count):
      self._add_constraints_american_port_singlearrival(k)
      self._add_constraints_american_port_singledeparture_with_arrival(k)
      self._add_constraints_vessels_must_have_cargo_space_for_each_unload(k)

    # Unnecessary constraints, (given the "cargo_space_for_each_unlock" constraint)
    # but they seem to help the model converge more quickly!
    if self.add_duplicate_constraints:
      self._add_vessel_capacity_constraints()
      for k in range(self.vessel_count):
        self._add_constraints_vessels_must_cross_pacific_togoto_american_ports(k)
        self._add_constraint_vessels_must_visit_port_to_unload_cargo(k)

  @property
  def asian_port_count(self):
    return len(self.asian_port_names)
  
  @property
  def american_port_count(self):
    return len(self.american_port_names)

  @property
  def port_names(self):
    return self.asian_port_names + self.american_port_names

  @property
  def port_count(self):
    return self.asian_port_count + self.american_port_count

  @property
  def vessel_count(self):
    return len(self.vessel_names)

  def _create_vessel_variables(self, k: int) -> dict:
    vessel_k_variables = {}
    
    ckj_cargo_unloading_variables = self.m.addVars(
      [
        f'c_{k}-{j+self.asian_port_count}'
        for j in range(self.american_port_count)
      ],
      vtype=GRB.INTEGER if self.enforce_integer_cargo else GRB.CONTINUOUS,
      lb=0,
      ub=self.vessel_capacities[k],
      obj=0,
      name=f"vessel-{k}-cargo"
    )
    vessel_k_variables["cargo_unloading"] = ckj_cargo_unloading_variables

    r_kj_cargo_remaining_variables = self.m.addVars(
      [
        f'r_{k}-{j+self.asian_port_count}'
        for j in range(self.american_port_count)
      ],
      vtype=GRB.INTEGER if self.enforce_integer_cargo else GRB.CONTINUOUS,
      lb=0,
      ub=self.vessel_capacities[k],
      obj=0,
      name=f"vessel_{k}_cargo_remaining"
    )
    vessel_k_variables["cargo_remaining"] = r_kj_cargo_remaining_variables

    xkij_transpacific_crossing_variables = self.m.addVars(
      [
        f'x_{k}_{i}_{self.asian_port_count+j}'
        for i in range(self.asian_port_count)
        for j in range(self.american_port_count)
        if self.vessel_origins[k] == i
      ],
      vtype=GRB.BINARY,
      lb=0,
      ub=1,
      obj=[
        self.port_distance_matrix[j, i] * self.vessel_costfactor[k]
        # f'x-crosstranspacific-ves{k}-port{i}-port{j}'
        for i in range(self.asian_port_count)
        for j in range(self.american_port_count)
        if self.vessel_origins[k] == i
      ],
      name=f"vessel-{k}-pacific-crossing"
    )
    vessel_k_variables["transpacific_crossing"] = xkij_transpacific_crossing_variables

    ykij_americancoast_transit_variables = self.m.addVars(
      [
        f'y_{k}-{j1 + self.asian_port_count}-{j2 + self.asian_port_count}'
        for j1 in range(self.american_port_count)
        for j2 in range(self.american_port_count)
        if j1 != j2
      ],
      vtype=GRB.BINARY,
      lb=0,
      ub=1,
      obj=[
        # f'x-crosstranspacific-ves{k}-port{i}-port{j}'
        self.port_distance_matrix[j1, self.asian_port_count + j2] * self.vessel_costfactor[k]
        for j1 in range(self.american_port_count)
        for j2 in range(self.american_port_count)
        if j1 != j2
      ],
      name=f"vessel-{k}-american-port-routes"
    )
    vessel_k_variables["american_routes"] = ykij_americancoast_transit_variables
    return vessel_k_variables

  def _add_vessel_capacity_constraints(self):
    # Vessel's cargo must not exceed capacity
    self.m.addConstrs(
      (
          sum([
            self.vessel_variables[k]["cargo_unloading"][f"c_{k}-{self.asian_port_count + j}"]
            for j in range(self.american_port_count)
          ]) <= self.vessel_capacities[k]
          for k in range(len(self.vessel_names)
        )
      ),
      name="vessel-capacity-constraints"
    )

  def _add_transpacific_singlecrossing_constraints(self):
    # Each vessel must cross the ocean only once - from its origin to 
    # some american west coast destination
    self.m.addConstrs(
      (
        sum(
          self.vessel_variables[k]["transpacific_crossing"][f"x_{k}_{self.vessel_origins[k]}_{self.asian_port_count + j}"]
          for j in range(self.american_port_count)
        ) <= 1
        for k in range(len(self.vessel_names))
      ),
      name="transpacific-single-crossing-constraints"
    )

  def _add_constraints_american_port_singlearrival(self, k: int):
    # Each vessel can only arrive at each port once
    self.m.addConstrs(
      (
        sum(
          # Routes from other american ports
          [
            self.vessel_variables[k]["american_routes"][f"y_{k}-{self.asian_port_count + j2}-{self.asian_port_count + j}"]
            for j2 in range(self.american_port_count)
            if j2 != j
          ]
          # Transpacific route from vessel origin
          + [
            self.vessel_variables[k]["transpacific_crossing"][f"x_{k}_{self.vessel_origins[k]}_{self.asian_port_count + j}"]
          ]
        ) <= 1
        for j in range(self.american_port_count)
      ),
      name=f"vessel-{k}-american-port-single-arrival-constraints"
    )

  def _add_constraints_american_port_singledeparture_with_arrival(self, k: int):
    # Each vessel can only leave a port once, and only if it has arrived
    self.m.addConstrs(
      (
        # Departing routes (departure sum <= arrivals == 0 or 1)
        sum(
          self.vessel_variables[k]["american_routes"][f"y_{k}-{self.asian_port_count + j}-{self.asian_port_count + j2}"]
          for j2 in range(self.american_port_count)
          if j != j2
        )
        <=

        # Incoming routes (sum = 0 or 1)
        sum(
          # Routes from other american ports
          [
            self.vessel_variables[k]["american_routes"][f"y_{k}-{self.asian_port_count + j2}-{self.asian_port_count + j}"]
            for j2 in range(self.american_port_count)
            if j2 != j
          ]
          # Transpacific route from vessel origin
          + [
            self.vessel_variables[k]["transpacific_crossing"][f"x_{k}_{self.vessel_origins[k]}_{self.asian_port_count + j}"]
          ]
        ) 
        for j in range(self.american_port_count)
      ),
      name=f"vessel-{k}-port-departure-requires-arrival"
    )
  
  def _add_constraints_vessels_must_cross_pacific_togoto_american_ports(self, k: int):
    # Vessels cannot do american port crossings unless they crossed the pacific
    self.m.addConstr(
      sum(
        self.vessel_variables[k]["american_routes"][f"y_{k}-{self.asian_port_count + j1}-{self.asian_port_count + j2}"]
        for j1 in range(self.american_port_count)
        for j2 in range(self.american_port_count)
        if j1 != j2
      ) <= 
      self.american_port_count *
      sum(
        self.vessel_variables[k]["transpacific_crossing"][f"x_{k}_{self.vessel_origins[k]}_{self.asian_port_count + j}"]
        for j in range(self.american_port_count)
      ),
      name=f'vessel-{k}-must-cross-pacific-to-go-to-american-ports'
    )

  def _add_constraint_vessels_must_visit_port_to_unload_cargo(self, k: int):
    # A vessel may not unload cargo at a port unless it has arrived there
    self.m.addConstrs(
      (
        self.vessel_variables[k]["cargo_unloading"][f"c_{k}-{self.asian_port_count + j}"]
        <= 
        self.vessel_capacities[k] *
        # incoming routes
        sum(
          [ # Routes from other american ports
            self.vessel_variables[k]["american_routes"][f"y_{k}-{self.asian_port_count + j2}-{self.asian_port_count + j}"]
            for j2 in range(self.american_port_count)
            if j2 != j
          ] # Transpacific route from vessel origin
          + [
            self.vessel_variables[k]["transpacific_crossing"][f"x_{k}_{self.vessel_origins[k]}_{self.asian_port_count + j}"]
          ]
        )
        for j in range(self.american_port_count)
      ),
      name=f"vessel-{k}-port-unloading-requires-arrival"
    )

  def _add_constraints_vessels_must_have_cargo_space_for_each_unload(self, k:int):
    # A vessel must have cargo remaining to unload
    self.m.addConstrs(
      (
        # Max cargo incoming
        sum(
          self.vessel_variables[k]["american_routes"][f"y_{k}-{self.asian_port_count + j_in}-{self.asian_port_count + j}"]
          * 
          self.vessel_variables[k]["cargo_remaining"][f"r_{k}-{self.asian_port_count + j_in}"]
          for j_in in range(self.american_port_count) if j_in != j
        ) +
          self.vessel_variables[k]["transpacific_crossing"][f"x_{k}_{self.vessel_origins[k]}_{self.asian_port_count + j}"] 
          *
          self.vessel_capacities[k]
        - 
          self.vessel_variables[k]["cargo_unloading"][f"c_{k}-{self.asian_port_count + j}"]
        == 
          self.vessel_variables[k]["cargo_remaining"][f"r_{k}-{self.asian_port_count + j}"]
        for j in range(self.american_port_count)
      ),
      name=f"vessel-{k}-cargo-remaining-to-unload"
    )

  def _add_constraint_cargo_demand_must_be_satisfied(self):
    # Inter-port cargo demand must be satisfied
    for i in range(self.asian_port_count):
      for j in range(self.american_port_count):
        self.m.addConstr(
          # All cargo unloaded by ships originating from port i to port j
          sum(
            [
              self.vessel_variables[k]["cargo_unloading"][f"c_{k}-{self.asian_port_count + j}"]
              for k in range(self.vessel_count)
              if self.vessel_origins[k] == i
            ]
          ) ==
          # Must be equal to the demand from port i to port j
          self.port_demand_matrix[j, i]          
        )

  def optimize(self, timelimit_secs: int = 20):
    self.m.Params.OutputFlag = 1 if self.enable_gurobi_logging else 0
    self.m.Params.timeLimit = timelimit_secs
    self.m.modelSense = GRB.MINIMIZE
    # print(f"Will optimize with timelimit of {timelimit_secs} seconds")
    self.m.optimize()
    print(f"Optimization complete. Gap: {self.m.MIPGap:.2%}")
    print(f"Status: {self.m.status}")
    total_teu = np.sum(self.port_demand_matrix)
    cost_per_teu = self.m.objVal / total_teu
    print(f"Cost: ${self.m.objVal/1000000:.2f} mil USD (${int(cost_per_teu)} per TEU)")
    return self.m.objVal, self.m.MIPGAP, cost_per_teu

  def print_status(self):
    
    for k in range(self.vessel_count):
      self.print_vessel_path(k)
    print()
    self.print_port_matrix()

  def print_port_matrix(self):
    print("".rjust(self.longest_asian_port_name), end="  ")

    total_col_justification = 6    
    print("Total".rjust(total_col_justification), end="  ")
    for j in range(self.american_port_count):
      print(self.american_port_names[j], end="  " if j < self.american_port_count - 1 else "\n")

    for i in range(self.asian_port_count):
      print(self.asian_port_names[i].rjust(self.longest_asian_port_name), end="  ")
      total_cargo = sum((
        self.vessel_variables[k]["cargo_unloading"][f"c_{k}-{self.asian_port_count + j}"].x
        for k in range(self.vessel_count)
        for j in range(self.american_port_count)
        if self.vessel_origins[k] == i
      ))
      print(str(int(total_cargo)).rjust(total_col_justification), end="  ")

      for j in range(self.american_port_count):
        port_name = self.american_port_names[j]
        total_cargo_to_portj = sum((
          self.vessel_variables[k]["cargo_unloading"][f"c_{k}-{self.asian_port_count + j}"].x
          for k in range(self.vessel_count)
          if self.vessel_origins[k] == i
        ))
        print(str(int(total_cargo_to_portj)).rjust(len(port_name)), end="  " if j < self.american_port_count - 1 else "\n")



  


  def get_vessel_results(self, k: int):
    route = self.get_vessel_route(k)
    route_names = [self.port_names[i] for i in route]
    
    def get_name(i: int): 
      return route_names[i] if len(route_names) > i else ""
    def get_cargo(i: int):
      if len(route) <= i or i == 0:
        return None
      port_i = route[i]
      return int(self.vessel_variables[k]["cargo_unloading"][f"c_{k}-{port_i}"].x)
    def get_distance(i: int):
      if len(route) <= i or i == 0:
        return None
      port_i1 = route[i - 1]
      port_i2 = route[i]
      return self.port_distance_matrix[port_i2 - self.asian_port_count, port_i1]

    return OutputVesselRouteRow(
      ship_name=self.vessel_names[k],
      start_port=self.port_names[self.vessel_origins[k]],
      capacity_teu=self.vessel_capacities[k],
      max_speed=self.vessel_maxspeeds[k],
      est_cost_per_nm_at_18knots=self.vessel_costfactor[k],
      port_1=get_name(1),
      cargo_1=get_cargo(1),
      dist_1=get_distance(1),
      port_2=get_name(2),
      cargo_2=get_cargo(2),
      dist_2=get_distance(2),
      port_3=get_name(3),
      cargo_3=get_cargo(3),
      dist_3=get_distance(3),
      port_4=get_name(4),
      cargo_4=get_cargo(4),
      dist_4=get_distance(4),
      port_5=get_name(5),
      cargo_5=get_cargo(5),
      dist_5=get_distance(5),
    )
    
  def get_vessel_route(self, k: int) -> List[int]:
    origin_port_index = self.vessel_origins[k]
    next_american_port = None
    route = [
      origin_port_index
    ]
    for j in range(self.american_port_count):
      if self.vessel_variables[k]['transpacific_crossing'][f'x_{k}_{origin_port_index}_{self.asian_port_count + j}'].x > 0:
        next_american_port = j
        break

    while next_american_port is not None:
      curr_port = next_american_port
      route.append(next_american_port + self.asian_port_count)
      next_american_port = None
      for j in range(self.american_port_count):
        if j != curr_port and self.vessel_variables[k]['american_routes'][f'y_{k}-{self.asian_port_count + curr_port}-{self.asian_port_count + j}'].x > 0:
          next_american_port = j
          break
    
    return route

  def print_vessel_path(self, k: int):
    origin_port_index = self.vessel_origins[k]
    origin_port_name = self.port_names[origin_port_index]
    name_w_quote = '"' + self.vessel_names[k] + '"'

    capacity = self.vessel_capacities[k]
    total_cargo_unloaded = np.sum(
      self.vessel_variables[k]['cargo_unloading'][f'c_{k}-{self.asian_port_count + j}'].x
      for j in range(self.american_port_count)
    )

    # print(f'  {int(total_cargo_unloaded)}/{int(capacity)} total cargo')
    remaining_cargo = int(total_cargo_unloaded)
    origin_port_label = f'{origin_port_name}:{remaining_cargo}:r{int(capacity)}'
    print(f'Vessel {k:< 2} {name_w_quote: ^{self.longest_vessel_name+2}}  {origin_port_label: <{self.longest_asian_port_name+13}}', end='')
    
    route = self.get_vessel_route(k)

    next_american_port = None
    for j in range(self.american_port_count):
      if self.vessel_variables[k]['transpacific_crossing'][f'x_{k}_{origin_port_index}_{self.asian_port_count + j}'].x > 0:
        next_american_port = j
        break
    
    # port_entry_pad_amount = self.longest_american_port_name+5
    if next_american_port is None:
      print(f"<nocross>", end='')

    self.vessel_variables[k]['cargo_unloading']

    visited_port_indexes = set()
    route_i = 1 # skip origin port
    while route_i < len(route):
      curr_american_port = route[route_i] - self.asian_port_count
      route_i += 1
      visited_port_indexes.add(curr_american_port)
      cargo_unloaded = self.vessel_variables[k]['cargo_unloading'][f'c_{k}-{self.asian_port_count + curr_american_port}'].x
      remaining_cargo -= int(cargo_unloaded)
      max_potential_cargo_remaining = self.vessel_variables[k]['cargo_remaining'][f'r_{k}-{self.asian_port_count + curr_american_port}'].x
      total_cargo_unloaded += cargo_unloaded
      labeltext = f'{self.american_port_names[curr_american_port]}:u{int(cargo_unloaded)}:h{remaining_cargo}:r{int(max_potential_cargo_remaining)}'
      print(f' --> {labeltext: ^{self.longest_american_port_name+10}}', end='')

    for j1 in range(self.american_port_count):
      for j2 in range(self.american_port_count):
        if j1 != j2:
          val = self.vessel_variables[k]['american_routes'][f'y_{k}-{j1 + self.asian_port_count}-{j2 + self.asian_port_count}'].x
          if val > 0 and j2 not in visited_port_indexes:
            cargo_dropped_off = self.vessel_variables[k]['cargo_unloading'][f'c_{k}-{j2 + self.asian_port_count}'].x
            print(f"\t\tUNEXPECTED ROUTE: {self.american_port_names[j1]} ->  {self.american_port_names[j2]} (dropped {cargo_dropped_off} cargo)")

    print(" ||")
    
if __name__ == "__main__":
  problem = TranspacificCargoRoutingProblem()
  problem.optimize()
  problem.print_status()

  print('Done')