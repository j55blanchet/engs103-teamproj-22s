#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Tuple
import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap, PowerNorm

from ..dataprep.port_loader import load_port_file

def plot_routes(routes: List[Tuple[Tuple[float, float],Tuple[float, float]]], us_ports: List[Tuple[float, float]], asian_ports: List[Tuple[float, float]], out_filename='flights_map_mpl.png', absolute=False):
    """Plots the given CSV data files use matplotlib basemap and saves it to
    a PNG file.

    Args:
        in_filename: Filename of the CSV containing the data points.
        out_filename: Output image filename
        color_mode: Use 'screen' if you intend to use the visualisation for
                    on screen display. Use 'print' to save the visualisation
                    with printer-friendly colors.
        absolute: set to True if you want coloring to depend on your dataset
                  parameter value (ie for comparison).
                  When set to false, each coordinate pair gets a different
                  color.
    """
    # define the expected CSV columns
    

    # num_routes = len(routes.index)

    # # normalize the dataset for color scale
    # norm = PowerNorm(0.3, routes['nb_flights'].min(),
    #                  routes['nb_flights'].max())
    # # norm = Normalize(routes['nb_flights'].min(), routes['nb_flights'].max())

    # # create a linear color scale with enough colors
    # if absolute:
    #     n = routes['nb_flights'].max()
    # else:
    #     n = num_routes
    # cmap = LinearSegmentedColormap.from_list('cmap_flights', color_list,
    #                                          N=n)
    # create the map and draw country boundaries
    plt.figure(figsize=(27, 20))
    m = Basemap(
        projection = 'cyl',
        llcrnrlon = 90, llcrnrlat = -60,
        urcrnrlon = 270, urcrnrlat = 70
    )

    coast_color = '#999999'
    bg_color = '#cccccc'
    blue = '#ccccff'
    white = '#ffffff'
    port_color = '#3333FF'
    route_color = '#ff0000'

    m.drawcoastlines(color=coast_color, linewidth=1.0)
    m.fillcontinents(color=bg_color, lake_color=blue)
    m.drawmapboundary(fill_color=white)

    for i, port in us_ports.iterrows():
        x, y = m(port['lon'], port['lat'])
        m.plot(x + 360, y, 'ro', markersize=5, color=port_color)
    for i, port in asian_ports.iterrows():
        m.plot(port['lon'], port['lat'], 'ro', markersize=5, color=port_color, latlon=True)


    # plot each route with its color depending on the number of flights #302.325
    for route in routes:
        # if absolute:
        #     color = cmap(norm(int(route['nb_flights'])))
        # else:
        #     color = cmap(i * 1.0 / num_routes)
        src, dst = route
        src_lat, src_lon = src
        dst_lat, dst_lon = dst

        lons = [(lon + 360) % 360 for lon in [src_lon, dst_lon]]
        x, y = m(lons, [src_lat, dst_lat])
        m.plot(x, y, color=route_color) # color=color, linewidth=1.0)

    # save the map
    plt.savefig(out_filename, format='png', bbox_inches='tight')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--asian_ports', type=str, default='source-data/asian_ports.csv')
    parser.add_argument('--us_ports', type=str, default='source-data/us_ports.csv')
    parser.add_argument('--optimized_routes', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    us_ports = pd.read_csv(args.us_ports, index_col=1)
    asian_ports = pd.read_csv(args.asian_ports, index_col=1)
    optimized_routes = pd.read_csv(args.optimized_routes)

    col_pairs = [
        ('start_port', 'port_1'),
        ('port_1', 'port_2'),
        ('port_2', 'port_3'),
        ('port_3', 'port_4'),
        ('port_4', 'port_5'),
    ]

    def get_port_coords(port_name: str):
        if port_name in us_ports.index:
            return us_ports.loc[port_name, ['lat', 'lon']].values
        elif port_name in asian_ports.index:
            return asian_ports.loc[port_name, ['lat', 'lon']].values
        else:
            return None


    routes = []

    for source, dest in col_pairs:
        for i, ship in optimized_routes.iterrows():
            source_port = ship[source]
            dest_port = ship[dest]
            if not isinstance (source_port, str) or not isinstance(dest_port, str):
                continue
            if len(source_port) > 0 and len(dest_port) > 0:
                source_coords = get_port_coords(source_port)
                dest_coords = get_port_coords(dest_port)
                if source_coords is not None and dest_coords is not None:
                    routes.append((source_coords, dest_coords))

    # routes = []
    # for key, count in route_counts.items():
    #     source_p, dest_p = key.split('^')
    #     lat, long = us_ports.loc[source_p, ['lat', 'lon']]
    #     routes.append([])


    # use 'screen' color mode for on-screen display. Use 'print' if you intend
    # to print the map
    plot_routes(routes, us_ports, asian_ports, args.output, absolute=False)
