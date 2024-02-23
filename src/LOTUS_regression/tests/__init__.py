from __future__ import annotations

from pathlib import Path


def load_example_data(filename):
    import os

    import pandas as pd

    file_path = Path(__file__).parent / "data" / filename

    data = pd.read_csv(file_path, parse_dates=True, index_col="time")

    return data[(data.index > "1978")]
