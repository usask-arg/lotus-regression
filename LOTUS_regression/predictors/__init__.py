from LOTUS_regression.predictors import download
import os
import pandas as pd


def load_data(filename):
    import pandas as pd
    import os

    file_path = os.path.join(os.path.dirname(__file__), 'data', filename)

    data = pd.read_csv(file_path, parse_dates=True, index_col='time')

    return data[(data.index > '1977')]


def remake_example_data():
    pred = pd.DataFrame()
    pred['enso'] = download.load_enso(2)

    pred['trop'] = download.load_trop(True)

    pred['solar'] = download.load_solar()

    pred['qboA'] = download.load_qbo(3)['pca']
    pred['qboB'] = download.load_qbo(3)['pcb']
    pred['qboC'] = download.load_qbo(3)['pcc']

    pred.index.name = 'time'

    pred -= pred.mean()
    pred /= pred.std()

    pred['linear_pre'] = download.load_linear(1997)['pre']
    pred['linear_post'] = download.load_linear(1997)['post']

    pred.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'predictors.csv'))

if __name__ == "__main__":
    remake_example_data()