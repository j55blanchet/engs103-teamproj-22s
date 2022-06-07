#%%
import matplotlib.pyplot as plt
from typing import List, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import random
from queue import PriorityQueue, SimpleQueue
import numpy.random
import math

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
    
    def dock_ship(self, t: float, ship: Ship, rng: np.random.Generator, dock_time_mean: float, dock_time_sd: float):
        self.curent_ship = ship

        expected_cargo_time = ship.capacity * self.days_per_teu
        randomized_cargo_time = rng.exponential(expected_cargo_time)
        randomized_dock_time = rng.normal(dock_time_mean, dock_time_sd)
        self.completion_date = t + randomized_cargo_time + randomized_dock_time

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
        nonfleet_cargo_utilization_normal: float = 0.3,
        nonfleet_cargo_utilization_sd: float = 0.1,
        docking_time_mean: float = 3.5 / 24.0, # 3.5 hrs

        docking_time_sd: float = 0.5 / 24.0, # 30 min
    ):
        ship_data = pd.read_csv(ship_data_filepath)
        terminal_data = pd.read_csv(terminal_data_filepath)
        port_data = pd.read_csv(port_data_filepath)

        self.event_counters = {}

        self.queue_length_over_time = []

        self.nonfleet_ship_counter = 0
        self.nonfleet_arrival_rate = nonfleet_arrival_rate
        self.nonfleet_length_mean = nonfleet_length_mean
        self.nonfleet_length_sd = nonfleet_length_sd
        self.nonfleet_teu_per_length_mean = nonfleet_teu_per_length_mean
        self.nonfleet_teu_per_length_sd = nonfleet_teu_per_length_sd
        self.nonfleet_cargo_utilization_normal = nonfleet_cargo_utilization_normal
        self.nonfleet_cargo_utilization_sd = nonfleet_cargo_utilization_sd

        self.docking_time_mean = docking_time_mean
        self.docking_time_sd = docking_time_sd

        self.rng = rng
        self.terminals = [
            Terminal(
                name=row["Name"],
                bearth_length=row["Berth"],
                # container_yard_area=row["Container Yard Area"],
                cranes=row["Total_Crane"],
                id=-1,
            )
            for i, row in 
            terminal_data[terminal_data["Port"] == "Long Beach"].iterrows()
        ]
        # Set indices appropiately
        for i in range(len(self.terminals)):
            self.terminals[i].id = i

        self._ships_in_transit = PriorityQueue()
        self._count_ships_in_transit = 0

        self.next_nonfleet_ship: Union[None, Tuple[float, Ship]] = None
        self.simulate_next_nonfleet_ship(cur_time=0.0)
        self.last_event = 0

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
        event = self.next_event
        if isinstance(self.next_event, ShipArrivalEvent):
            if self.next_event.is_fleet_ship:
                self.event_counters["fleet_ship_arrives"] = self.event_counters.get("fleet_ship_arrives", 0) + 1
            else:
                self.event_counters["nonfleet_ship_arrives"] = self.event_counters.get("nonfleet_ship_arrives", 0) + 1
            self.handle_ship_arrival(self.next_event)
            
        elif isinstance(self.next_event, CargoProcessedEvent):
            self.handle_cargo_processed(self.next_event)
            self.event_counters["cargo_processed"] = self.event_counters.get("cargo_processed", 0) + 1
        self.last_event = event

        self.queue_length_over_time.append(
            (event.timestamp,len(self.ships_in_queue))
        )

    def simulate_next_nonfleet_ship(self, cur_time: float):
        self.nonfleet_ship_counter += 1
        
        PERCENT_CARGO_THAT_IS_IN_OUR_FLEET = 0.7
        percent_cargo_on_nonfleet = 1 - PERCENT_CARGO_THAT_IS_IN_OUR_FLEET
        time_to_arrival_adjustment_ratio = 1 / percent_cargo_on_nonfleet

        time_to_arrival = time_to_arrival_adjustment_ratio * self.rng.exponential(1 / self.nonfleet_arrival_rate)
        ship_length = self.rng.normal(self.nonfleet_length_mean, self.nonfleet_length_sd)
        ship_teu = ship_length * self.rng.normal(self.nonfleet_teu_per_length_mean, self.nonfleet_teu_per_length_sd)
        cargo_utilization = max(0.05, self.rng.normal(self.nonfleet_cargo_utilization_normal, self.nonfleet_cargo_utilization_sd))

        ship = Ship(
            name = f"Non-fleet ship {self.nonfleet_ship_counter}",
            capacity = ship_teu,
            roundtrip_mean = math.inf,
            roundtrip_sd=0,
            longbeach_cargo = cargo_utilization * ship_teu,
            is_fleet_ship = False
        )
        self.next_nonfleet_ship = (cur_time + time_to_arrival, ship)

    def handle_ship_arrival(self, event: ShipArrivalEvent):

        ship: Union[None, Ship] = None
        if event.is_fleet_ship:
            ship = self._ships_in_transit.get()[1]
            self._count_ships_in_transit -= 1
        else:
            # nonfleet ship
            _, ship = self.next_nonfleet_ship
            self.simulate_next_nonfleet_ship(event.timestamp)

        # Case 1 - if space at a terminal, assign ship directly to terminal
        for terminal in self.terminals:
            if terminal.can_fit_ship(ship):
                terminal.dock_ship(event.timestamp, ship, self.rng, self.docking_time_mean, self.docking_time_sd)
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
                terminal.dock_ship(event.timestamp, ship, self.rng, self.docking_time_mean, self.docking_time_sd)
                self.ships_in_queue.pop(i)
                return

    def print_status(self):

        print(f"{len(self.ships_in_queue)} ships in queue")
        print(f"{self._count_ships_in_transit} ships in transit")
        print("Terminal Status:")
        for terminal in self.terminals:
            term_status = "empty" if terminal.curent_ship is None else terminal.curent_ship.name
            print(f"\t{terminal.name}: {term_status}", end='')
            finishes = terminal.completion_date
            print(f'\t{finishes=}')
        print(f"Last event: {self.last_event}")
        print(f"Next event: {self.next_event}")
        if len(self.event_counters) > 0:
            print("Events:")
        for key, value in self.event_counters.items():
            print(f"\t{key}: {value}")

    def print_queue_length_over_time(self):

        
        plt.plot([d[0] for d in self.queue_length_over_time], [d[1] for d in self.queue_length_over_time])
        mean_queue_length = np.mean(self.queue_length_over_time, axis=0)[1]
        plt.plot([self.queue_length_over_time[0][0], self.queue_length_over_time[-1][0]], [mean_queue_length, mean_queue_length], label=f"Mean queue length: {mean_queue_length:.2}")
        plt.xlabel('Days')
        plt.ylabel('Queue length')
        plt.title('Queue length over time')
        plt.legend()
        plt.savefig('queue_length_over_time.pdf')
        # plt.show()

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

    # print()
    # simulation.run(1000)
    # print()
    # simulation.print_status()
    
    print()
    simulation.run(10000)
    print()
    simulation.print_status()

    simulation.print_queue_length_over_time()


# %%
if __name__ == "__main__":
    main()
