from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def SBUV_NOAA(folder):
    vmr_files = Path.glob(Path(folder) / "*_mn*_vmr.dat")

    data = xr.concat([_parse_SBUV_NOAA(f) for f in vmr_files], dim="time")

    data = data.isel(time=data["time"].argsort())

    return data.where(data["vmr"] != 99)


def _parse_SBUV_NOAA(filename):
    year = re.search(r"_mn(\d*)_", filename).group(1)

    time = pd.date_range(
        datetime(year=int(year), month=1, day=1), periods=12, freq="MS"
    )

    pressure_levels = [0.5, 0.7, 1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50]

    lat_bins = np.arange(-87.5, 90, 5)

    vmr_values = np.ones((len(time), len(lat_bins), len(pressure_levels))) * np.nan
    num_in_bin = np.ones((len(time), len(lat_bins))) * np.nan

    data = xr.Dataset(
        {
            "vmr": (("time", "mean_latitude", "pressure"), vmr_values),
            "num": (("time", "mean_latitude"), num_in_bin),
        },
        coords={"pressure": pressure_levels, "mean_latitude": lat_bins, "time": time},
    )

    time_index = 0
    lat_index = 0

    with Path.open(filename) as f:
        for _i, line in enumerate(csv.reader(f, delimiter=" ", skipinitialspace=True)):
            if len(line) == 2:
                # Could be either (year, month) or (latitude, num_in_bin)
                if float(line[0]) > 1000:
                    # It is year
                    time_index = int(line[1]) - 1
                else:
                    # Is it latitude
                    lat_index = np.argmin(np.abs(float(line[0]) - lat_bins))
                    data.num.to_numpy()[time_index, lat_index] = int(line[1])
            else:
                # Data field, first line has 8 elements second has 7
                if len(line) == 8:
                    data.vmr.to_numpy()[time_index, lat_index, :8] = [
                        float(lin) for lin in line
                    ]
                elif len(line) == 7:
                    data.vmr.to_numpy()[time_index, lat_index, 8:] = [
                        float(lin) for lin in line
                    ]

    return data


if __name__ == "__main__":
    # _parse_SBUV_NOAA(r'X:/data/SBUV_NOAA/n09_v8_mn1996_vmr.dat')

    SBUV_NOAA(r"X:/data/SBUV_NOAA/")
