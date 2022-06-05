import argparse
import numpy as np

from ..dataprep.load_vessels import load_vessels_file
from ..dataprep.distance_table_loader import load_distance_table
from .optimization_model import OutputVesselRouteRow, TranspacificCargoRoutingProblem
    
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
    parser.add_argument('--distance-table-file', type=str, required=True)
    parser.add_argument('--vessels-file', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    args = parser.parse_args()

    asian_port_names = "Busan,Hakodate,Ho Chi Minh,Singapore,Hong Kong,Kobe,Manila,Melbourne,Shanghai,Kelang,Laem Chabang,Kaohsiung".split(',')
    american_port_names = "Los Angeles,Long Beach,Oakland,Seattle-Tacoma,Vancouver".split(',')

    distance_table = load_distance_table(args.distance_table_file)
    distance_matrix = np.array([
        [
            distance_table[american_port_name][asian_port_name]
            for asian_port_name in asian_port_names
        ] + 
        [
            0 if other_american_port_name == american_port_name else distance_table[american_port_name][other_american_port_name]
            for other_american_port_name in american_port_names
        ]
        for american_port_name in american_port_names
    ])

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

    vessels = load_vessels_file(args.vessels_file)

    origin_port_index_lookup = dict(
        (port, i) for i, port in enumerate(asian_port_names)
    )

    hyperparm_matrix = []
    hyperparm_matrix.append(["runtime", "integer_cargo", "extra_constraints", "total_cost", "MIPGap", "cost_per_teu"])
    for time in [1, 5, 10, 20, 40]:
        for enforce_integer_cargo, add_duplicate_constraints in [
            (False, False), (False, True), (True, False), (True, True)
        ]:
            print()
            print()
            print(f"{time=}s {enforce_integer_cargo=}, {add_duplicate_constraints=}")
            problem = TranspacificCargoRoutingProblem(
                asian_port_names=asian_port_names,
                american_port_names=american_port_names,
                port_distance_matrix=distance_matrix,
                port_demand_matrix=port_demand_matrix,

                vessel_names=[v.name for v in vessels],
                vessel_origins=[origin_port_index_lookup[v.origin_port] for v in vessels],
                vessel_capacities=[v.capacity_teu for v in vessels],
                vessel_maxspeeds=[v.max_speed_knots for v in vessels],
                vessel_costfactor=[v.cost_factor for v in vessels],

                enforce_integer_cargo=enforce_integer_cargo,
                add_duplicate_constraints=add_duplicate_constraints,
                enable_gurobi_logging=False
            )
            total_cost, mip_gap, cost_per_teu = problem.optimize(
                timelimit_secs= time
            )
            hyperparm_matrix.append([
                f"{float(time):.1f}".rjust(len(hyperparm_matrix[0][0])), 
                f"{enforce_integer_cargo}".rjust(len(hyperparm_matrix[0][1])), 
                f"{add_duplicate_constraints}".rjust(len(hyperparm_matrix[0][2])),
                f"{int(total_cost)}".rjust(len(hyperparm_matrix[0][3])),
                f"{float(mip_gap):.2%}".rjust(len(hyperparm_matrix[0][4])),
                f"{int(cost_per_teu)}".rjust(len(hyperparm_matrix[0][5])),
            ])
    
    print()
    print()
    for row in hyperparm_matrix:
        print("\t".join(row))

    print()
    print("==== DOING FULL RUN NOW ====")
    print()

    problem = TranspacificCargoRoutingProblem(
        asian_port_names=asian_port_names,
        american_port_names=american_port_names,
        port_distance_matrix=distance_matrix,
        port_demand_matrix=port_demand_matrix,

        vessel_names=[v.name for v in vessels],
        vessel_origins=[origin_port_index_lookup[v.origin_port] for v in vessels],
        vessel_capacities=[v.capacity_teu for v in vessels],
        vessel_maxspeeds=[v.max_speed_knots for v in vessels],
        vessel_costfactor=[v.cost_factor for v in vessels],

        enforce_integer_cargo=True
    )
    problem.optimize(
        timelimit_secs= 60 * 60
    )
    problem.print_status()

    print('Done. Writing routes csv')

    import csv
    with open(args.output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(OutputVesselRouteRow.csv_header_row())
        for k in range(len(vessels)):
            results = problem.get_vessel_results(k)
            writer.writerow(results.csv_row())
        

    

