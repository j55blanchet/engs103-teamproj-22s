from dataclasses import dataclass

@dataclass
class Vessel:
    name: str
    origin_port: str
    capacity_teu: int
    max_speed_knots: int
    cost_factor: float
    
def load_vessels_file(filepath: str):
    import csv
    with open(filepath) as f:
        reader = csv.reader(f)
        next(reader)
        vessels = [
            Vessel(
                name=row[0].strip(),
                origin_port=row[1].strip(),
                capacity_teu=int(row[2].strip()),
                max_speed_knots=int(row[3].strip()),
                cost_factor=float(row[4].strip())
            )
            for row in reader
            if len(row) > 1
        ]
        return vessels