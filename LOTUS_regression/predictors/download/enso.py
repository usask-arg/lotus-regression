import pandas as pd


def load_enso():
    data = pd.read_table('http://www.esrl.noaa.gov/psd/enso/mei/table.html', skiprows=12, skipfooter=41, sep='\s+',
                         index_col=0, engine='python')
    assert (data.index[0] == 1950)
    data = data.stack(dropna=True)
    data.index = pd.date_range(start='1950', periods=len(data), freq='M').to_period()
    return data

if __name__ == "__main__":
    print(load_enso())
