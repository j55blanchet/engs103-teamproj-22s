from dataclasses import dataclass

@dataclass
class Port:
    country: str = ''
    port_name: str = ''
    size: float = 0.0
    lat: float = 0.0
    long: float = 0.0

def load_port_file(filepath: str):
    import csv
    with open(filepath) as f:
        reader = csv.reader(f)
        next(reader) # skip header row
        ports = [
            Port(country=row[0].strip(), port_name=row[1].strip(), size=float(row[2].strip()), lat=float(row[3].strip()), long=float(row[4].strip()))
            for row in reader
            if len(row) >= 5 and not row[0].startswith('#')
        ]
        return ports