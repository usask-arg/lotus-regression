import numpy as np
import pandas as pd


def load_linear():
    num_months = 12 * (pd.datetime.now().year - 1979) + pd.datetime.now().month
    index = pd.date_range('1980-01', periods=num_months, freq='M').to_period(freq='M')
    pre = 1/120*pd.Series([t - 12 * (1997 - 1980) if t < 12 * (1997 - 1980) else 0 for t in range(num_months)], index=index,
                    name='pre')
    post = 1/120*pd.Series([t - 12 * (1997 - 1980) if t > 12 * (1997 - 1980) else 0 for t in range(num_months)], index=index,
                     name='post')
    return pd.concat([pre, post], axis=1)


if __name__ == "__main__":
    print(load_linear())
