# engs103-teamproj-22s
ENGS 103 (Operations Research) Team Project

## Installation

`python3 -m venv .env`
`source .env/bin/activate`
`pip install dataclass_csv`

## Data Preparation

**Port List**  

Top 10 Asian Ports sourced from <https://www.icontainers.com/us/2021/03/15/top-10-ports-asia/>
For Taiwan: <https://www.google.com/search?client=firefox-b-1-d&q=biggest+port+in+taiwan>
* List of port
* Longitude and Latitude
* Port Size (TEU) - used for import/export gravity model

Top West Coast Ports sourced from <https://www.pmsaship.com/wp-content/uploads/2021/06/West-Coast-Trade-Report-June-2021.pdf>

US Import Partners
<https://www.census.gov/foreign-trade/statistics/highlights/toppartners.html>

**Cargo Demand**: Transport demand between pairs of ports

Table Format:
| Port (Asia) | Port (West Coast)  | Demand (TEU) |
|:------------|:-------------------|:-------------|
| Shanghai    | Los Angeles        |  5           |
| Tokyo       | Los Angeles        |  3           |
| Soeul       | Tacoma             |  1           |
| Hong Kong   | Los Angeles        |  2           |




- Source / destination port pairs  
- Cargo quantity (weekly)

**Ports (Distance, GPS coordinate)**:
The distance table is based on the distance calculated in Pub. 151 published by National Geosptial-Intelligence Agency.
* Source port: Asian ports
* Sink port: US west coast ports
* GPS coordinate: Latitude and Longitude based on degree, i.e., W 127-36 = -127.6 deg
* Distance: Nautial mile (NM)

    
**Ships (Supply)**: List of ships  
The fleet is based on one of largest container shipping alliance ```Ocean Aliance``` consisting of
CMA CGM, COSCO, Evergreen. The total number of ships in ```Ocean Alliance``` are counted as 1258 (https://ajot.com/premium/ajot-ocean-carrier-alliances-the-tripartite)
* Extracted ship fleet calling US west coast ports: 88 (https://www.cma-cgm.com/ebusiness/schedules/port)
* Total TEU: 890,742 


**Ports (Queue Capacity)**: List of ports with assoc. info
