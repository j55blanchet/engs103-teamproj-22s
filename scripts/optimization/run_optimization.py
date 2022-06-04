import argparse
import numpy as np

from ..dataprep.vessel_fleet_loader import load_fleet_file
from ..dataprep.distance_table_loader import load_distance_table
from .optimization_model import TranspacificCargoRoutingProblem
    
def load_port_demand_matrix(filepath: str):
    import csv
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        port_demand_matrix = np.array([
            [int(row[i]) for i in range(1, len(row))]
            for row in reader
        ])
        return port_demand_matrix


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port-demand-matrix', type=str, required=True)
    parser.add_argument('--distances', type=str, required=True)
    parser.add_argument('--fleet', type=str, required=True)
    args = parser.parse_args()

    asian_port_names = "Busan,Hakodate,Ho Chi Minh,Singapore,Hong Kong,Kobe,Manila,Melbourne,Shanghai,Kelang,Laem Chabang,Kaohsiung".split(',')
    american_port_names = "Los Angeles,Long Beach,Oakland,Seattle-Tacoma,Vancouver".split(',')

    distance_table = load_distance_table(args.distances)
    distance_matrix = np.array([
        [
            distance_table[american_port_name][asian_port_name]
            for asian_port_name in asian_port_names
        ]
        for american_port_name in american_port_names
    ])

    fleet = load_fleet_file(args.fleet)

    port_demand_matrix = load_port_demand_matrix(
        filepath=args.port_demand_matrix
    )
    # np.array([
    #     [47790,7716,27043,49758,49660,7771,12945,1851,107276,18176,15469,19013],
    #     [39674,6406,22450,41308,41226,6451,10747,1536,89058,15089,12842,15784],
    #     [10047,1663,5818,10053,10276,1670,2583,349,23077,3664,3119,3921],
    #     [15251,2577,8454,14537,15198,2517,4314,421,34571,5285,4500,5826],
    #     [21871,3691,12149,20896,21824,3610,5585,607,49609,7599,6469,8364]
    # ])

    problem = TranspacificCargoRoutingProblem(
        asian_port_names=asian_port_names,
        american_port_names=american_port_names,
        port_distance_matrix=distance_matrix,
        port_demand_matrix=port_demand_matrix,

        # TODO: 
    )

    problem.optimize()
    problem.print_status()

    print('Done')

