import csv
from dataclasses import dataclass

@dataclass
class FleetEntry:
    name: str
    operator: str
    type: str
    geared: bool
    nominal_capaity_teu: int
    reefer_plugs: int
    deadweight: int
    year_built: int
    flag: str
    speed_knots: int

    @property
    def cost_factor(self):
        # FROM: https://www.researchgate.net/profile/Jan-Fransoo-2/publication/239853065_Ocean_container_transportan_underestimated_and_critical_link_in_global_supply_chain_performance/links/5459f1a10cf2bccc4912e69b/Ocean-container-transportan-underestimated-and-critical-link-in-global-supply-chain-performance.pdf?origin=publication_detail
        # $52 / nm for a 5000 TEU ship at 18 knots
        #  + an additional $1.9667 / nm per TEU
        #  Equates to $13266 base + 1.9667 * TEU
        # Let's assume that ships travel at 18 knots and are 90% full
        return 13266 + 1.9667 * self.nominal_capaity_teu * 0.9
    

def load_fleet_file(filepath: str):
    fleet:list[FleetEntry] = []
    with open(filepath) as fleet_csv:
        reader = csv.reader(fleet_csv)
        next(reader)
        for rowi, row in enumerate(reader):
            try:
                fleet.append(FleetEntry(
                    name=row[0].strip(),
                    operator=row[1].strip(),
                    type=row[2].strip(),
                    geared=row[3].strip() == 'Y',
                    nominal_capaity_teu=int(row[4].strip()),
                    reefer_plugs=int(row[5].strip()),
                    deadweight=int(row[6].strip()),
                    year_built=int(row[7].strip()),
                    flag=row[8].strip(),
                    speed_knots=int(row[9].strip())
                ))
            except Exception as e:
                print(f'Error on row {rowi}: {e}  ({row=})')
                pass
    return fleet