from dataclasses import dataclass

@dataclass
class Port:
    country: str = ''
    port_name: str = ''
    size: float = 0.0

def load_port_file(filepath: str):
    import csv
    with open(filepath) as f:
        reader = csv.reader(f)
        next(reader) # skip header row
        ports = [
            Port(country=row[0].strip(), port_name=row[1].strip(), size=float(row[2].strip()))
            for row in reader
            if len(row) >= 3 and not row[0].startswith('#')
        ]
        return ports