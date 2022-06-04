import csv
from dataclasses import dataclass

@dataclass
class DistanceTableEntry:
    destinationPort: str = ''
    destinationLatitude: float = 0.0
    destinationLongitude: float = 0.0
    sourcePort: str = ''
    sourceCountry: str = ''
    distanceInNauticalMiles: float = 0.0

def load_distance_table(distance_table_filepath: str):
    distance_table: dict[str, dict[str, float]] = {}
    with open(distance_table_filepath) as distance_table_csv:
        reader = csv.reader(distance_table_csv)
        next(reader) # skip header row
        for row in reader:
            entry = DistanceTableEntry(
                destinationPort=row[0].strip(),
                destinationLatitude=row[1].strip(),
                destinationLongitude=row[2].strip(),
                sourcePort=row[3].strip(),
                sourceCountry=row[4].strip(),
                distanceInNauticalMiles=float(row[5].strip())
            )
            distance_table[entry.destinationPort] = distance_table.get(entry.destinationPort, {})
            distance_table[entry.destinationPort][entry.sourcePort] = float(entry.distanceInNauticalMiles)

    return distance_table