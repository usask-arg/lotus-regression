Monthly zonal averages appear one file per year.

Mixing ratio files are organized as:
Year, month
Zone center1, # data days in zonal average
Profile data (15 for mixing ratio)
Zone center2, # data days in zonal average
Profile data (15 for mixing ratio)
etc


Monthly Dobson layer files are organized as:
Year, month
Zone center1, # data days in zonal average, 99.9, 99.9, total_ozone
Profile data (13 layers)
 Zone center2, # data days in zonal average, 99.9, 99.9, total_ozone
Profile data (13 layers)
etc


Total ozone
Missing data indicator for mixing ratio profile data is:99.
Missing data indicator for total_ozone is 999.9
Missing data indicator for mixing ratio profile data is:999.

Zones are every 5 degrees -85 to 85.:
Dobson layer bottoms are: 'surf', '64', '41', '25', '16',  '10', '6', '4', '2.5', '1.6',  '1', '.64', '.41'
Pressure levels are '.5', '.7', '1', '1.5', '2',  '3', '4', '5', '7', '10',  '15', '20', '30', '40', '50'
