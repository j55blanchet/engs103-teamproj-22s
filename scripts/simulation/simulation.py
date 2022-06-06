#%%

from typing import List, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import random
from queue import PriorityQueue, SimpleQueue
import numpy.random

@dataclass
class Event:
    timestamp: float

@dataclass
class ShipArrivalEvent(Event):
    is_fleet_ship: bool

@dataclass
class CargoProcessedEvent(Event):
    terminal_id: int

@dataclass
class Ship:
    name: str
    capacity: int # in TEU
    roundtrip_mean: float # in days
    roundtrip_sd: float # in days
    longbeach_cargo: int # in TEU
    is_fleet_ship: bool = True

    @property
    def length(self):
        if self.capacity > 14_400:
            return 1300
        elif self.capacity > 10_000:
            return 1200
        elif self.capacity > 5000:
            return 1100
        else:
            return 965  

@dataclass
class Terminal:
    name: str
    # port: str
    bearth_length: float # in feet
    # container_yard_area: float # in acres
    cranes: int

    id: int

    curent_ship: Union[Ship, None] = None

    completion_date: Union[float, None] = None

    def can_fit_ship(self, ship: Ship):
        return self.curent_ship is None and self.bearth_length >= ship.length
    
    def dock_ship(self, t: float, ship: Ship, rng: np.random.Generator):
        self.curent_ship = ship

        expected_cargo_time = ship.capacity * self.days_per_teu
        randomized_cargo_time = rng.exponential(expected_cargo_time)
        self.completion_date = t + randomized_cargo_time

    @property
    def next_event(self):
        return None if self.curent_ship is None else CargoProcessedEvent(self.completion_date, self.id)

    @property
    def days_per_teu(self):
        CRANES_MOVES_PER_HOUR = 26.5
        TEU_PER_CRANE_MOVE = 1.54
        HOURS_PER_DAY = 24
        return 1 / (self.cranes * CRANES_MOVES_PER_HOUR * TEU_PER_CRANE_MOVE * HOURS_PER_DAY)

@dataclass
class Port:
    name: str
    terminals: List[Terminal] = field(default_factory=list)
    queuing_ships: List[Ship] = field(default_factory=list)

class Simulation:

    def __init__(
        self,
        ship_data_filepath,
        terminal_data_filepath,
        port_data_filepath,
        rng = numpy.random.default_rng(),
        nonfleet_arrival_rate: float = 24.0,
        nonfleet_length_mean: float = 1000.0,
        nonfleet_length_sd: float = 100.0,
        nonfleet_teu_per_length_mean: float = 5.5,
        nonfleet_teu_per_length_sd: float = 0.5,
    ):
        ship_data = pd.read_csv(ship_data_filepath)
        terminal_data = pd.read_csv(terminal_data_filepath)
        port_data = pd.read_csv(port_data_filepath)

        self.nonfleet_ship_counter = 0
        self.nonfleet_arrival_rate = nonfleet_arrival_rate
        self.nonfleet_length_mean = nonfleet_length_mean
        self.nonfleet_length_sd = nonfleet_length_sd
        self.nonfleet_teu_per_length_mean = nonfleet_teu_per_length_mean
        self.nonfleet_teu_per_length_sd = nonfleet_teu_per_length_sd

        self.rng = rng

        self.last_event_t = 0

        self.terminals = [
            Terminal(
                name=row["Name"],
                bearth_length=row["Berth"],
                # container_yard_area=row["Container Yard Area"],
                cranes=1, #row["Total_Crane"],
                id=-1,
            )
            for i, row in 
            terminal_data[terminal_data["Port"] == "Long Beach"].iterrows()
        ]
        # Set indices appropiately
        for i in range(len(self.terminals)):
            self.terminals[i].id = i

        self.terminals = [
            self.terminals[0]
        ]

        self._ships_in_transit = PriorityQueue()
        self._count_ships_in_transit = 0

        self.next_nonfleet_ship: Union[None, Tuple[float, Ship]] = None

        for i, row in ship_data.iterrows():

            # Skip invalid ships
            if row["mean"] < 0:
                print(f"Skipping ship {i} '{row['shipname']}' with negative mean")
                continue

            ship = Ship(
                name = row["shipname"],
                capacity = row["teu"],
                roundtrip_mean = row["mean"],
                roundtrip_sd = row["stddev"],
                longbeach_cargo = row["longbeach_cargo"]
            )
            
            first_trip_length = rng.normal(ship.roundtrip_mean, ship.roundtrip_sd)
            percent_done_w_trip = rng.random()
            day_of_arrival = first_trip_length * percent_done_w_trip
            self.set_ship_to_sea(ship, day_of_arrival)
    
        self.ships_in_queue = []
        print("Done initializing")
        print()

    @property
    def next_ship_arrival_event(self):
        if self._ships_in_transit.empty():
            return None

        # Remove nearest ship from the PriorityQueue to read its arrival time
        #   *Don't forget to put it back in!*
        timestamp, arrival_ship = self._ships_in_transit.get()
        self._ships_in_transit.put((timestamp, arrival_ship))

        # Compare against next non-fleet ship. Return the one that arrives sooner
        if self.next_nonfleet_ship is not None:
            if self.next_nonfleet_ship[0] < timestamp:
                return ShipArrivalEvent(self.next_nonfleet_ship[0], is_fleet_ship=False)

        return ShipArrivalEvent(timestamp, is_fleet_ship=True)
    
    def set_ship_to_sea(self, ship: Ship, eta: float):
        # Don't need to simulate nonfleet ship's transit times
        if not ship.is_fleet_ship:
            return

        self._ships_in_transit.put((eta, ship))
        self._count_ships_in_transit += 1

    def print_status(self):

        print(f"{len(self.ships_in_queue)} ships in queue")
        print(f"{self._count_ships_in_transit} ships in transit")
        print("Terminal Status:")
        for terminal in self.terminals:
            term_status = "empty" if terminal.curent_ship is None else terminal.curent_ship.name
            print(f"\t{terminal.name}: {term_status}")
        print(f"Next event: {self.next_event}")
    
    @property
    def next_event(self):
        next_event = self.next_ship_arrival_event
        for terminal in self.terminals:
            terminal_event = terminal.next_event
            if next_event is None:
                next_event = terminal_event
            elif terminal_event is not None and terminal_event.timestamp < next_event.timestamp:
                next_event = terminal_event
        return next_event

    def update(self):
        if isinstance(self.next_event, ShipArrivalEvent):
            self.handle_ship_arrival(self.next_event)
        elif isinstance(self.next_event, CargoProcessedEvent):
            self.handle_cargo_processed(self.next_event)

    def simulate_next_nonfleet_ship(self, cur_time: float):
        self.nonfleet_ship_counter += 1
        
        time_to_arrival = self.rng.exponential

        pass

    def handle_ship_arrival(self, event: ShipArrivalEvent):

        ship: Union[None, Ship] = None
        if event.is_fleet_ship:
            ship = self._ships_in_transit.get()[1]
            self._count_ships_in_transit -= 1
        else:
            # nonfleet ship
            _, ship = self.next_nonfleet_ship
            self.simulate_next_nonfleet_ship()

        # Case 1 - if space at a terminal, assign ship directly to terminal
        for terminal in self.terminals:
            if terminal.can_fit_ship(ship):
                terminal.dock_ship(event.timestamp, ship, self.rng)
                return

        # Case 2 - no space at a terminal, enqueue ship
        self.ships_in_queue.append(ship)

    def handle_cargo_processed(self, event: CargoProcessedEvent):
        terminal = self.terminals[event.terminal_id]
        
        undocked_ship = terminal.curent_ship
        voyage_time = self.rng.normal(undocked_ship.roundtrip_mean, undocked_ship.roundtrip_sd)
        self.set_ship_to_sea(undocked_ship, event.timestamp + voyage_time)

        terminal.curent_ship = None
        terminal.completion_date = None

        for i in range(len(self.ships_in_queue)):
            ship = self.ships_in_queue[i]
            if terminal.can_fit_ship(ship):
                terminal.dock_ship(event.timestamp, ship, self.rng)
                self.ships_in_queue.pop(i)
                return

    def run(self, event_count: int):
        print(f"Running simulation for {event_count} events")
        for _ in range(event_count):
            self.update()
        
#%%
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ship_data_filepath", type=str, required=True)
    parser.add_argument("--terminal_data_filepath", type=str, required=True)
    parser.add_argument("--port_data_filepath", type=str, required=True)
    args = parser.parse_args()

    simulation = Simulation(
        ship_data_filepath=args.ship_data_filepath,
        terminal_data_filepath=args.terminal_data_filepath,
        port_data_filepath=args.port_data_filepath
    )
    print()
    simulation.print_status()

    simulation.run(1000)

    simulation.print_status()




# %%
if __name__ == "__main__":
    main()
