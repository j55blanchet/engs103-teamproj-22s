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


    
**Ships (Supply)**: List of ships  
- Ship


**Ports (Queue Capacity)**: List of ports with assoc. info
