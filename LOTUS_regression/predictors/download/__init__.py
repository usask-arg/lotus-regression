import numpy as np
import pandas as pd
import requests
import re
import ftplib
import xarray as xr
import os
import appdirs
import time
from datetime import datetime
from io import StringIO


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
    Downloads the ENSO from https://www.esrl.noaa.gov/psd/enso/mei/data/meiv2.data

    Parameters
    ----------
    lag_months : int, Optional. Default 0
        The numbers of months of lag to introduce to the ENSO signal
    """
    data = pd.read_table('https://www.esrl.noaa.gov/psd/enso/mei/data/meiv2.data', skiprows=1, skipfooter=4, sep='\s+',
                         index_col=0, engine='python', header=None)
    assert (data.index[0] == 1979)
    data = data.stack()
    data = data[data > -998]
    data.index = pd.date_range(start='1979', periods=len(data), freq='M').to_period()

    data = data.shift(lag_months)

    return data


def load_linear(inflection=1997):
    """
    Returns two piecewise linear components with a given inflection point in value / decade.

    Parameters
    ----------
    inflection : int, Optional. Default 1997
    """
    start_year = 1974

    num_months = 12 * (pd.datetime.now().year - start_year) + pd.datetime.now().month
    index = pd.date_range('1975-01', periods=num_months, freq='M').to_period(freq='M')
    pre = 1/120*pd.Series([t - 12 * (inflection - (start_year+1)) if t < 12 * (inflection - (start_year+1)) else 0 for t in range(num_months)], index=index,
                    name='pre')
    post = 1/120*pd.Series([t - 12 * (inflection - (start_year+1)) if t > 12 * (inflection - (start_year+1)) else 0 for t in range(num_months)], index=index,
                     name='post')
    return pd.concat([pre, post], axis=1)


def load_independent_linear(pre_trend_end='1997-01-01', post_trend_start='2000-01-01'):
    """
    Creates the predictors required for performing independent linear trends.

    Parameters
    ----------
    pre_trend_end: str, Optional. Default '1997-01-01'

    post_trend_start: str, Optional.  Default '2000-01-01'
    """
    NS_IN_YEAR = float(31556952000000000)

    start_year = 1974

    num_months = 12 * (pd.datetime.now().year - start_year) + pd.datetime.now().month
    index = pd.date_range('1975-01', periods=num_months, freq='M').to_period(freq='M')

    pre_delta = -1*(index.to_timestamp() - pd.to_datetime(pre_trend_end)).values
    post_delta = (index.to_timestamp() - pd.to_datetime(post_trend_start)).values

    assert(pre_delta.dtype == np.dtype('<m8[ns]'))
    assert(post_delta.dtype == np.dtype('<m8[ns]'))

    pre_delta = pre_delta.astype(np.int64) / NS_IN_YEAR
    post_delta = post_delta.astype(np.int64) / NS_IN_YEAR

    pre_const = np.ones_like(pre_delta)
    pre_const[pre_delta < 0] = 0

    post_const = np.ones_like(post_delta)
    post_const[post_delta < 0] = 0

    # Check if we need a gap constant
    pre_plus_post = pre_const + post_const
    if np.any(pre_plus_post == 0):
        need_gap_constant = True

        gap_constant = np.ones_like(pre_plus_post)
        gap_constant[pre_plus_post == 1] = 0

        gap_linear = np.ones_like(pre_plus_post)
        gap_linear[pre_plus_post == 1] = 0

        lt = (index.to_timestamp() - pd.to_datetime(pre_trend_end)).values
        lt = lt.astype(np.int64) / NS_IN_YEAR
        gap_linear *= lt

        gap_constant = pd.Series(gap_constant, index=index, name='gap_const')
        gap_linear = pd.Series(gap_linear, index=index, name='gap_linear')
    else:
        need_gap_constant = False

    pre_delta[pre_delta < 0] = 0
    post_delta[post_delta < 0] = 0

    pre = pd.Series(-1*pre_delta / 10, index=index, name='pre')
    post = pd.Series(post_delta / 10, index=index, name='post')

    post_const = pd.Series(post_const, index=index, name='post_const')
    pre_const = pd.Series(pre_const, index=index, name='pre_const')

    if need_gap_constant:
        data = pd.concat([pre, post, post_const, pre_const, gap_constant, gap_linear], axis=1)
    else:
        data = pd.concat([pre, post, post_const, pre_const], axis=1)

    return data


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

    data = pd.read_fwf(StringIO(requests.get('http://www.geo.fu-berlin.de/met/ag/strat/produkte/qbo/qbo.dat').text),
                       skiprows=200, header=None,
                       colspecs=[(0, 5), (6, 10), (12, 16), (19, 23), (26, 30), (33, 37), (40, 44), (47, 51), (54, 58)],
                         delim_whitespace=True, index_col=1, parse_dates=True, date_parser=date_parser,
                         names=['station', 'month', '70', '50', '40', '30', '20', '15', '10'])
    data.index = data.index.to_period(freq='M')

    data.drop('station', axis=1, inplace=True)
    data = data[:-1]

    if pca > 0:
        from string import ascii_lowercase
        pca_d = decomp.PCA(n_components=pca)
        for idx, c in zip(range(pca), ascii_lowercase):
            data['pc' + c] = pca_d.fit_transform(data.values).T[idx, :]

    return data


def load_solar():
    """
    Gets the solar F10.7 from 'http://www.spaceweather.ca/data-donnee/sol_flux/sx-5-mavg-eng.php'.
    """
    sess = requests.session()
    sess.get('https://omniweb.gsfc.nasa.gov/')

    today = datetime.today()

    page = sess.get('https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi?activity=retrieve&res=daily&spacecraft=omni2_daily&start_date=19631128&end_date={}&vars=50&scale=Linear&ymin=&ymax=&charsize=&symsize=0.5&symbol=0&imagex=640&imagey=480'.format(today.strftime('%Y%M%d')))

    # Won't have data for today, find the largest possible range
    last_day = page.text[page.text.rindex('19631128 - ') + 11:page.text.rindex('19631128 - ') + 8 + 11]

    page = sess.get('https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi?activity=retrieve&res=daily&spacecraft=omni2_daily&start_date=19631128&end_date={}&vars=50&scale=Linear&ymin=&ymax=&charsize=&symsize=0.5&symbol=0&imagex=640&imagey=480'.format(last_day))

    data = StringIO(page.text[page.text.rindex('YEAR'):page.text.rindex('<hr>')])

    solar = pd.read_csv(data, delimiter='\s+')
    solar = solar[:-1]

    solar['dt'] = pd.to_datetime((solar['YEAR'].astype('int') * 1000) + solar['DOY'].astype(int), format='%Y%j')
    solar = solar.set_index(keys='dt')
    solar = solar.where(solar['1'] != 999.9)
    solar = solar.resample('MS').mean()

    return solar['1'].rename('f10.7').to_period(freq='M')


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

    # Find the last non-zero entry and extend to the current date
    last_nonzero_idx = data[data['tau'] != 0].index[-1]
    last_nonzero_idx = np.argmax(data.index == last_nonzero_idx)

    # Extend the index to approximately now
    num_months = 12 * (pd.datetime.now().year - data.index[0].year) + pd.datetime.now().month
    index = pd.date_range(data.index[0].to_timestamp(), periods=num_months, freq='M').to_period(freq='M')

    # New values
    vals = np.zeros(len(index))
    vals[:last_nonzero_idx] = data['tau'].values[:last_nonzero_idx]
    vals[last_nonzero_idx:] = data['tau'].values[last_nonzero_idx]

    new_aod = pd.Series(vals, index=index, name='aod')

    return new_aod


def load_glossac_aod():
    data = xr.open_dataset(r'X:/data/sasktran/GloSSAC_V2_CF.nc')

    times = data.time.values
    years = times // 100
    months = times % 100

    # Extend the index to approximately now
    num_months = 12 * (pd.datetime.now().year - years[0]) + pd.datetime.now().month
    index = pd.date_range(pd.to_datetime(datetime(year=years[0], month=months[0], day=1)), periods=num_months, freq='M').to_period(freq='M')

    aod = data.sel(wavelengths_glossac=525)['Glossac_Aerosol_Optical_Depth'].values
    latitudes = data.lat.values
    integration_weights = np.cos(np.deg2rad(latitudes))
    integration_weights /= np.nansum(integration_weights)

    aod = np.trapz(aod * integration_weights[np.newaxis, :], axis=1)

    extended_aod = np.zeros(len(index))
    extended_aod[:len(aod)] = aod
    extended_aod[len(aod):] = aod[-1]

    aod_df = pd.Series(extended_aod, index=index, name='aod')

    return aod_df


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

if __name__ == "__main__":
    load_solar()