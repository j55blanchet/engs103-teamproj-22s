
from dataclasses import dataclass
from typing import List
import csv
import numpy as np

from .distance_table_loader import load_distance_table
from .port_loader import load_port_file
    
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--asian-ports', type=str, required=True)
    parser.add_argument('--us-ports', type=str, required=True)
    parser.add_argument('--distance-table', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    asian_ports = load_port_file(args.asian_ports)
    us_ports = load_port_file(args.us_ports)
    # us_port_names = set((port.port_name for port in us_ports))

    distance_table = load_distance_table(args.distance_table)
            
    grav_weights = np.array([
        [
            (us_port.size * asian_port.size) / (distance_table[us_port.port_name][asian_port.port_name]**2)
            for us_port in us_ports      
        ]
        for asian_port in asian_ports
    ])

    cargo_demands_list = []
    for i, asian_port in enumerate(asian_ports):
        total_gravity = np.sum(grav_weights[i])
        cargo_demands = [
            int(asian_port.size * grav_weights[i, j] / total_gravity)
            for j, us_port in enumerate(us_ports)
        ]
        cargo_demands_list.append(cargo_demands)
        
    with open(args.output, 'w') as output_csv:
        writer = csv.writer(output_csv)
        writer.writerow(['us_port'] + [port.port_name for port in asian_ports])

        for us_port_index, row in enumerate(np.array(cargo_demands_list).T):
            writer.writerow([us_ports[us_port_index].port_name] + list(row))
    

if __name__ == "__main__":
    main()