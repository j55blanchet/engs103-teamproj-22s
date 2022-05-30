
from dataclasses import dataclass
from typing import List
import csv

@dataclass
class Port:
    country: str = ''
    port_name: str = ''
    size: float = 0.0

@dataclass
class TradePartner:
    country: str = ''
    trade_volume: float = float('nan')
    
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--asian-ports', type=str, required=True)
    parser.add_argument('--us-ports', type=str, required=True)
    parser.add_argument('--trade-partners', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)

    args = parser.parse_args()

    asian_ports: List[Port] = []
    with open(args.asian_ports) as asianports_csv:
        reader = csv.reader(asianports_csv)
        next(reader) # skip header row
        asian_ports = [
            Port(country=row[0], port_name=row[1], size=float(row[2]))
            for row in reader
        ]
    asian_port_total_capacity_sq = sum((port.size**2 for port in asian_ports))
    
   
    us_ports: List[Port] = []
    with open(args.us_ports) as usports_csv:
        reader = csv.reader(usports_csv)
        next(reader)
        us_ports = [
            Port(country=row[0], port_name=row[1], size=float(row[2]))
            for row in reader
        ]
    us_port_total_capacity_sq = sum((port.size**2 for port in us_ports))

    trade_partners: List[TradePartner] = []
    with open(args.trade_partners) as trade_partners_csv:
        reader = csv.reader(trade_partners_csv)
        next(reader)
        trade_partners = [
            TradePartner(country=row[0], trade_volume=float(row[1]))
            for row in reader
        ]


    with open(args.output, 'w') as output_csv:
        writer = csv.writer(output_csv)
        writer.writerow([''] + [port.port_name for port in us_ports])
        for port in asian_ports:
            for trade_partner in trade_partners:
                asian_port_demand = trade_partner.trade_volume / asian_port_total_capacity_sq
                us_port_demand = trade_partner.trade_volume / us_port_total_capacity_sq
                writer.writerow([port.port_name, trade_partner.country, asian_port_demand, us_port_demand])


if __name__ == "__main__":
    main()