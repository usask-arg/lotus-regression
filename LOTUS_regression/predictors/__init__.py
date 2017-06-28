from LOTUS_regression.predictors import download
import os


def load_data(filename):
    import pandas as pd
    import os

    file_path = os.path.join(os.path.dirname(__file__), 'data', filename)

    data = pd.read_csv(file_path, parse_dates=True, index_col='time')

    return data[(data.index > '1977')]


def remake_example_data():
    pred = download.load_linear(1997)

    pred['enso'] = download.load_enso(2)

    pred['trop'] = download.load_trop(True)

    pred['solar'] = download.load_solar()

    pred['qbo_pcaA'] = download.load_qbo(3)['pca']
    pred['qbo_pcaB'] = download.load_qbo(3)['pcb']
    pred['qbo_pcaC'] = download.load_qbo(3)['pcc']

    pred.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'predictors.csv'))

if __name__ == "__main__":
    remake_example_data()