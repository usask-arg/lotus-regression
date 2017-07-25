from LOTUS_regression.predictors import download
import os
import pandas as pd
import numpy as np


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


def make_baseline_pwlt():
    pred = pd.DataFrame()
    pred['enso'] = download.load_enso(0)
    pred['trop'] = download.load_trop(True)
    pred['solar'] = download.load_solar()
    pred['qboA'] = download.load_qbo(2)['pca']
    pred['qboB'] = download.load_qbo(2)['pcb']
    pred['aod'] = download.load_giss_aod()

    pred.index.name = 'time'

    pred -= pred.mean()
    pred /= pred.std()

    pred['linear_pre'] = download.load_linear(1997)['pre']
    pred['linear_post'] = download.load_linear(1997)['post']

    pred['constant'] = np.ones(len(pred.index))

    pred.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'pred_baseline_pwlt.csv'))


def make_baseline_ilt():
    pred = pd.DataFrame()
    pred['enso'] = download.load_enso(0)
    pred['trop'] = download.load_trop(True)
    pred['solar'] = download.load_solar()
    pred['qboA'] = download.load_qbo(2)['pca']
    pred['qboB'] = download.load_qbo(2)['pcb']
    pred['aod'] = download.load_giss_aod()

    pred.index.name = 'time'

    pred -= pred.mean()
    pred /= pred.std()

    linear = download.load_independent_linear(pre_trend_end='1997-01-01', post_trend_start='2000-01-01')

    pred['linear_pre'] = linear['pre']
    pred['linear_post'] = linear['post']
    pred['pre_const'] = linear['pre_const']
    pred['post_const'] = linear['post_const']
    pred['gap_cons'] = linear['gap_const']

    pred.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'pred_baseline_ilt.csv'))


def make_baseline_eesc():
    pred = pd.DataFrame()
    pred['enso'] = download.load_enso(0)
    pred['trop'] = download.load_trop(True)
    pred['solar'] = download.load_solar()
    pred['qboA'] = download.load_qbo(2)['pca']
    pred['qboB'] = download.load_qbo(2)['pcb']
    pred['aod'] = download.load_giss_aod()

    pred.index.name = 'time'

    pred -= pred.mean()
    pred /= pred.std()

    pred[['eesc_1', 'eesc_2']] = download.load_orthogonal_eesc('Z:/data/LOTUS/proxies/EESC_Damadeo/eesc.txt')[['eesc_1', 'eesc_2']]

    pred['constant'] = np.ones(len(pred.index))

    pred.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'pred_baseline_eesc.csv'))

if __name__ == "__main__":
    make_baseline_pwlt()
    make_baseline_ilt()
    make_baseline_eesc()
    remake_example_data()
