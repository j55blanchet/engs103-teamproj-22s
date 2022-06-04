import csv
from random import shuffle, choices
from .vessel_fleet_loader import load_fleet_file
from .port_loader import load_port_file
from .load_vessels import Vessel

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--asian-ports-file', type=str, required=True)
    parser.add_argument('--fleet-file', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    fleet = load_fleet_file(args.fleet_file)
    asian_ports = load_port_file(args.asian_ports_file)

    # Randomize fleet order
    shuffle(fleet)

    ports_with_unmet_demand = [port.port_name for port in asian_ports]
    unmet_port_demand = dict(
        (port.port_name, port.size)
        for port in asian_ports
    ) 

    # First make sure all ports have their demand met
    vessels: list[Vessel] = []
    while len(ports_with_unmet_demand) > 0:
        
        if len(fleet) == 0:
            raise Exception('Ran out of vessels in the fleet')

        ship = fleet.pop()
        chosen_port = None       
        
        port = ports_with_unmet_demand[-1]
        unmet_port_demand[port] -= ship.nominal_capaity_teu
        chosen_port = port
        if unmet_port_demand[port] <= 0:
            ports_with_unmet_demand.pop()
        
        vessel = Vessel(
            name=ship.name,
            origin_port=chosen_port,
            capacity_teu=ship.nominal_capaity_teu,
            max_speed_knots=ship.speed_knots,
            cost_factor=ship.cost_factor
        )
        vessels.append(vessel)

    # Now randomly assign the rest of the ships 
    chosen_ports = choices(
        [port.port_name for port in asian_ports], 
        [port.size for port in asian_ports],
        k=len(fleet)
    )

    while len(fleet) > 0:
        ship = fleet.pop()
        vessel = Vessel(
            name=ship.name,
            origin_port=chosen_ports.pop(),
            capacity_teu=ship.nominal_capaity_teu,
            max_speed_knots=ship.speed_knots,
            cost_factor=ship.cost_factor
        )
        vessels.append(vessel)
    
    with open(args.output, 'w') as output_csv:
        writer = csv.writer(output_csv)
        writer.writerow([
            'name',
            'origin_port',
            'capacity_teu',
            'max_speed_knots',
            'cost_factor'
        ])

        for vessel in vessels:
            writer.writerow([
                vessel.name,
                vessel.origin_port,
                vessel.capacity_teu,
                vessel.max_speed_knots,
                vessel.cost_factor
            ])








if __name__ == "__main__":
    main()