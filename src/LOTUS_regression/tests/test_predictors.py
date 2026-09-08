from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from LOTUS_regression.predictors import download


def test_load_linear_ends_at_current_month():
    original_to_datetime = pd.to_datetime

    def frozen_to_datetime(value, *args, **kwargs):
        if value == "today":
            return pd.Timestamp("2024-06-15")
        return original_to_datetime(value, *args, **kwargs)

    with patch.object(download.pd, "to_datetime", side_effect=frozen_to_datetime):
        linear = download.load_linear()

    assert linear.index[0] == pd.Period("1975-01", freq="M")
    assert linear.index[-1] == pd.Period("2024-06", freq="M")
    assert len(linear) == 12 * (2024 - 1975) + 6
