from __future__ import annotations

import numpy as np


def add_seasonal_components(basis_df, num_components):
    for column in basis_df:
        n_harmonic = num_components.get(column, 0)

        for i in range(n_harmonic):
            basis_df[column + "_sin" + str(i)] = basis_df[column] * np.sin(
                2 * np.pi * (basis_df.index.dayofyear - 1) / 365.25 * (i + 1)
            )
            basis_df[column + "_cos" + str(i)] = basis_df[column] * np.cos(
                2 * np.pi * (basis_df.index.dayofyear - 1) / 365.25 * (i + 1)
            )

    return basis_df
