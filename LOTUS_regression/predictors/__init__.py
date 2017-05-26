def load_data(filename):
    import pandas as pd
    import os

    file_path = os.path.join(os.path.dirname(__file__), 'data', filename)

    data = pd.read_csv(file_path, parse_dates=True, index_col='time')

    return data[(data.index > '1978') & (data.index < '2017')]
