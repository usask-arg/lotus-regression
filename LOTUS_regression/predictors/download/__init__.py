import numpy as np
import pandas as pd
import requests
import re
import ftplib
import xarray as xr
import os
import appdirs
import time


def load_eesc():
    """
    Calculates an EESC from the polynomial values [9.451393e-10, -1.434144e-7, 8.5901032e-6, -0.0002567041,
    0.0040246245, -0.03355533, 0.14525718, 0.71710218, 0.1809734]
    """
    poly = [9.451393e-10, -1.434144e-7, 8.5901032e-6, -0.0002567041,
            0.0040246245, -0.03355533, 0.14525718, 0.71710218, 0.1809734]
    np.polyval(poly, 1)

    num_months = 12 * (pd.datetime.now().year - 1979) + pd.datetime.now().month
    index = pd.date_range('1979-01', periods=num_months, freq='M').to_period(freq='M')
    return pd.Series([np.polyval(poly, month/12) for month in range(num_months)], index=index)


def load_enso(lag_months=0):
    """
    Downloads the ENSO from http://www.esrl.noaa.gov/psd/enso/mei/table.html

    Parameters
    ----------
    lag_months : int, Optional. Default 0
        The numbers of months of lag to introduce to the ENSO signal
    """
    data = pd.read_table('http://www.esrl.noaa.gov/psd/enso/mei/table.html', skiprows=12, skipfooter=41, sep='\s+',
                         index_col=0, engine='python')
    assert (data.index[0] == 1950)
    data = data.stack(dropna=True)
    data.index = pd.date_range(start='1950', periods=len(data), freq='M').to_period()

    data = data.shift(lag_months)

    return data


def load_linear(inflection=1997):
    """
    Returns two piecewise linear components with a given inflection point in value / decade.

    Parameters
    ----------
    inflection : int, Optional. Default 1997
    """
    num_months = 12 * (pd.datetime.now().year - 1974) + pd.datetime.now().month
    index = pd.date_range('1975-01', periods=num_months, freq='M').to_period(freq='M')
    pre = 1/120*pd.Series([t - 12 * (inflection - 1975) if t < 12 * (inflection - 1975) else 0 for t in range(num_months)], index=index,
                    name='pre')
    post = 1/120*pd.Series([t - 12 * (inflection - 1975) if t > 12 * (inflection - 1975) else 0 for t in range(num_months)], index=index,
                     name='post')
    return pd.concat([pre, post], axis=1)


def load_qbo(pca=3):
    """
    Loads the QBO from http://www.geo.fu-berlin.de/met/ag/strat/produkte/qbo/qbo.dat.  If pca is set to an integer (default 3) then
    that many principal components are taken.  If pca is set to 0 then the raw QBO data is returned.

    Parameters
    ----------
    pca : int, optional.  Default 3.
    """
    import sklearn.decomposition as decomp
    # yymm date parser
    def date_parser(s):
        s = int(s)
        return pd.datetime(2000 + s // 100 if (s // 100) < 50 else 1900 + s // 100, s % 100, 1)

    data = pd.read_fwf('http://www.geo.fu-berlin.de/met/ag/strat/produkte/qbo/qbo.dat',
                       skiprows=200, header=None,
                       colspecs=[(0, 5), (6, 10), (12, 16), (19, 23), (26, 30), (33, 37), (40, 44), (47, 51), (54, 58)],
                         delim_whitespace=True, index_col=1, parse_dates=True, date_parser=date_parser,
                         names=['station', 'month', '70', '50', '40', '30', '20', '15', '10'])
    data.index = data.index.to_period(freq='M')

    data.drop('station', axis=1, inplace=True)

    if pca > 0:
        from string import ascii_lowercase
        pca_d = decomp.PCA(n_components=3)
        for idx, c in zip(range(pca), ascii_lowercase):
            data['pc' + c] = pca_d.fit_transform(data.values).T[idx, :]

    return data

def load_solar():
    """
    Gets the solar F10.7 from 'http://www.spaceweather.ca/data-donnee/sol_flux/sx-5-mavg-eng.php'.
    """
    page = requests.get('http://www.spaceweather.ca/data-donnee/sol_flux/sx-5-mavg-eng.php')

    start = page.text.rindex('Absolute Flux') + len('Absolute Flux')
    end = page.text.index('</table>', start)

    text = page.text[start:end]

    text = re.sub("[^0-9.]", " ", text)
    values = text.split()
    table = [values[5 * i:5 * (i + 1)] for i in range(len(values) // 5)]

    solar = pd.DataFrame([[int(row[0]), int(row[1]), float(row[4])] for row in table],
                         columns=['year', 'month', 'f10.7'])

    solar['dt'] = solar.apply(lambda row: pd.datetime(int(row.year), int(row.month), 1), axis=1).dt.to_period(freq='M')

    return solar.set_index(keys='dt')['f10.7']


def load_trop(deseasonalize=True):
    """
    Gets the tropical tropopause pressure from ftp.cdc.noaa.gov.  The tropical tropopause pressure is automatically
    deseasonalized by default to remove the strong seasonal cycle.

    Parameters
    ----------
    deseasonalize : bool, optional.  Default True
        If set to false deseasonalization will not be done.
    """
    path = 'Datasets/ncep.reanalysis.derived/tropopause/'
    filename = 'pres.tropp.mon.mean.nc'

    save_path = os.path.join(appdirs.user_data_dir(), filename)
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Only fetch from the ftp if the file does not exist or is greater than one week out of date.
    if not os.path.exists(save_path) or time.time() - os.path.getmtime(save_path) > 60*60*24*7:
        ftp = ftplib.FTP("ftp.cdc.noaa.gov")
        ftp.login()
        ftp.cwd(path)
        ftp.retrbinary("RETR " + filename, open(save_path, 'wb').write)
        ftp.quit()

    data = xr.open_dataset(save_path)

    trop_only = data.pres.mean(dim='lon').where((-5 < data.lat) & (data.lat < 5)).mean(dim='lat')

    if deseasonalize:
        anom = trop_only.groupby('time.month') - trop_only.groupby('time.month').mean(dim='time')
    else:
        anom = trop_only

    return anom.to_dataframe('pres').pres.to_period(freq='M')


def load_ao():
    """
    Loads the arctic oscillation index from ncep
    """
    data = pd.read_table('http://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii',
                         delim_whitespace=True,
                         header=None,
                         names=['year', 'month', 'ao'])

    data['dt'] = data.apply(lambda row: pd.datetime(int(row.year), int(row.month), 1), axis=1).dt.to_period(freq='M')

    return data.set_index(keys='dt')['ao']


def load_aao():
    data = pd.read_table('http://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/aao/monthly.aao.index.b79.current.ascii',
                         delim_whitespace=True,
                         header=None,
                         names=['year', 'month', 'aao'])

    data['dt'] = data.apply(lambda row: pd.datetime(int(row.year), int(row.month), 1), axis=1).dt.to_period(freq='M')

    return data.set_index(keys='dt')['aao']


def load_nao():
    """
    Loads the north atlantic oscillation index from noaa
    :return:
    """
    data = pd.read_table(
        'http://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii',
        delim_whitespace=True,
        header=None,
        names=['year', 'month', 'nao'])

    data['dt'] = data.apply(lambda row: pd.datetime(int(row.year), int(row.month), 1), axis=1).dt.to_period(
        freq='M')

    return data.set_index(keys='dt')['nao']


def load_ehf(filename):
    """
    Loads the eddy heat flux data from the file erai_ehf_monthly_1978_2016.txt provided on the LOTUS ftp server in
    the folder Proxies-Weber
    """
    data = pd.read_table(filename, delim_whitespace=True, header=None, skiprows=4, names=['year', 'month', 'sh_ehf', 'nh_ehf'])

    data['dt'] = data.apply(lambda row: pd.datetime(int(np.floor(row.year)), int(row.month), 1), axis=1).dt.to_period(
        freq='M')

    data = data.drop(['year', 'month'], axis=1)

    return data.set_index(keys='dt')


def load_giss_aod():
    """
    Loads the giss aod index from giss
    """
    filename = 'tau_map_2012-12.nc'

    save_path = os.path.join(appdirs.user_data_dir(), filename)
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Only fetch from the ftp if the file does not exist
    if not os.path.exists(save_path) or time.time():
        r = requests.get(r'https://data.giss.nasa.gov/modelforce/strataer/tau_map_2012-12.nc')

        with open(save_path, 'wb') as f:
            f.write(r.content)

    data = xr.open_dataset(save_path)

    data = data.mean(dim='lat')['tau'].to_dataframe()

    data.index = data.index.map(lambda row: pd.datetime(int(row.year), int(row.month), 1)).to_period(freq='M')
    data.index.names = ['time']
    return data['tau']


def load_solar_mg2():
    """
    Loads the bremen solar composite mg2 index
    """
    data = pd.read_table(
        'http://www.iup.uni-bremen.de/gome/solar/MgII_composite.dat',
        delim_whitespace=True,
        header=22,
        names=['year', 'month', 'day', 'index', 'error', 'id'],
        parse_dates={'time': [0, 1, 2]},
        index_col='time'
        )

    return data.resample('1M').mean().to_period(freq='M')['index']


def load_orthogonal_eesc(filename):
    """
    Calculates two orthogonal eesc terms from the predicted eesc at 6 different ages of air, uses the EESC.txt
    datafile from the LOTUS ftp server in the folder EESC_Damadeo
    """
    data = pd.read_table(filename, delim_whitespace=True, header=3)

    import sklearn.decomposition as decomp

    pca = decomp.PCA(n_components=2)
    data['eesc_1'], data['eesc_2'] = pca.fit_transform(data.values).T

    def frac_year_to_datetime(start):
        from datetime import datetime, timedelta

        year = int(start)
        rem = start - year

        base = datetime(year, 1, 1)
        result = base + timedelta(seconds=(base.replace(year=base.year + 1) - base).total_seconds() * rem)

        return result

    data.index = data.index.map(frac_year_to_datetime)

    data = data.resample('MS').interpolate('linear')

    data.index = data.index.to_period(freq='M')

    data = data.drop(['1.0', '2.0', '3.0', '4.0', '5.0', '6.0'], axis=1)

    data /= data.std()

    return data
