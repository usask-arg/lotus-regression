import pandas as pd
import sklearn.decomposition as decomp


def load_qbo(load_pca=True):
    # yymm date parser
    def date_parser(s):
        s = int(s)
        return pd.datetime(2000 + s // 100 if (s // 100) < 50 else 1900 + s // 100, s % 100, 1)

    # Line 381 is the beginning of 1984.
    # Starting here makes parsing the file much, much easier.
    data = pd.read_table('http://www.geo.fu-berlin.de/met/ag/strat/produkte/qbo/qbo.dat', skiprows=381, header=None,
                         delim_whitespace=True, index_col=1, parse_dates=True, date_parser=date_parser,
                         names=['station', 'month', '70', '50', '40', '30', '20', '15', '10'])
    data.index = data.index.to_period(freq='M')
    assert(data.index[0] == pd.Period('1984-01', 'M'))

    data.drop('station', axis=1, inplace=True)

    if load_pca:
        pca = decomp.PCA(n_components=3)
        data['pca'], data['pcb'], data['pcc'] = pca.fit_transform(data.values).T

    return data


if __name__ == "__main__":
    print(load_qbo())
