from dataclasses import dataclass

@dataclass
class Vessel:
    name: str
    origin_port: str
    capacity_teu: int
    max_speed_knots: int
    cost_factor: float
    