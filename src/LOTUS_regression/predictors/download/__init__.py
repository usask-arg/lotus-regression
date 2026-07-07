from __future__ import annotations

import ftplib
import time
from pathlib import Path

import appdirs
import numpy as np
import pandas as pd
import requests
import xarray as xr


def _current_month_period():
    today = pd.to_datetime("today")
    return pd.Period(year=today.year, month=today.month, freq="M")


def _is_netcdf_file(path):
    with Path.open(path, "rb") as f:
        magic = f.read(4)
    return magic.startswith(b"CDF") or magic == b"\x89HDF"


def _download_netcdf_if_needed(url, save_path):
    directory = save_path.parent
    if not directory.exists():
        directory.mkdir(parents=True)

    if save_path.exists() and _is_netcdf_file(save_path):
        return

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    if not (response.content.startswith(b"CDF") or response.content[:4] == b"\x89HDF"):
        msg = f"Downloaded data from {url} is not a NetCDF file"
        raise ValueError(msg)

    with Path.open(save_path, "wb") as f:
        f.write(response.content)


def _timedelta_years(delta):
    if not np.issubdtype(delta.dtype, np.timedelta64):
        msg = f"Expected timedelta values, got {delta.dtype}"
        raise TypeError(msg)
    return delta.astype("timedelta64[ns]").astype(np.int64) / float(31556952000000000)


def load_eesc():
    """
    Calculates an EESC from the polynomial values [9.451393e-10, -1.434144e-7, 8.5901032e-6, -0.0002567041,
    0.0040246245, -0.03355533, 0.14525718, 0.71710218, 0.1809734]
    """
    poly = [
        9.451393e-10,
        -1.434144e-7,
        8.5901032e-6,
        -0.0002567041,
        0.0040246245,
        -0.03355533,
        0.14525718,
        0.71710218,
        0.1809734,
    ]
    np.polyval(poly, 1)

    num_months = (
        12 * (pd.to_datetime("today").year - 1979) + pd.to_datetime("today").month
    )
    index = pd.period_range("1979-01", periods=num_months, freq="M")
    return pd.Series(
        [np.polyval(poly, month / 12) for month in range(num_months)], index=index
    )


def load_enso(lag_months=0):
    """
    Downloads the ENSO from https://www.esrl.noaa.gov/psd/enso/mei/data/meiv2.data

    Parameters
    ----------
    lag_months : int, Optional. Default 0
        The numbers of months of lag to introduce to the ENSO signal
    """
    data = pd.read_table(
        "https://www.esrl.noaa.gov/psd/enso/mei/data/meiv2.data",
        skiprows=1,
        skipfooter=4,
        sep=r"\s+",
        index_col=0,
        engine="python",
        header=None,
    )
    assert data.index[0] == 1979
    data = data.to_numpy().flatten()
    data = data[data > -998]

    data = pd.DataFrame(
        data, index=pd.period_range(start="1979-01", periods=len(data), freq="M")
    )

    return data.shift(lag_months)


def load_linear(inflection=1997):
    """
    Returns two piecewise linear components with a given inflection point in value / decade.

    Parameters
    ----------
    inflection : int, Optional. Default 1997
    """
    start = pd.Period("1975-01", freq="M")
    today = pd.to_datetime("today")
    end = pd.Period(year=today.year, month=today.month, freq="M")
    index = pd.period_range(start=start, end=end, freq="M")

    t = np.arange(len(index))
    inflection_index = pd.Period(f"{inflection}-01", freq="M").ordinal - start.ordinal

    pre = pd.Series(np.minimum(t - inflection_index, 0) / 120, index=index, name="pre")
    post = pd.Series(
        np.maximum(t - inflection_index, 0) / 120, index=index, name="post"
    )
    return pd.concat([pre, post], axis=1)


def load_independent_linear(pre_trend_end="1997-01-01", post_trend_start="2000-01-01"):
    """
    Creates the predictors required for performing independent linear trends.

    Parameters
    ----------
    pre_trend_end: str, Optional. Default '1997-01-01'

    post_trend_start: str, Optional.  Default '2000-01-01'
    """
    index = pd.period_range("1975-01", end=_current_month_period(), freq="M")

    pre_delta = -1 * (index.to_timestamp() - pd.to_datetime(pre_trend_end)).to_numpy()
    post_delta = (index.to_timestamp() - pd.to_datetime(post_trend_start)).to_numpy()

    pre_delta = _timedelta_years(pre_delta)
    post_delta = _timedelta_years(post_delta)

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

        lt = (index.to_timestamp() - pd.to_datetime(pre_trend_end)).to_numpy()
        lt = _timedelta_years(lt)
        gap_linear *= lt

        gap_constant = pd.Series(gap_constant, index=index, name="gap_const")
        gap_linear = pd.Series(gap_linear, index=index, name="gap_linear")
    else:
        need_gap_constant = False

    pre_delta[pre_delta < 0] = 0
    post_delta[post_delta < 0] = 0

    pre = pd.Series(-1 * pre_delta / 10, index=index, name="pre")
    post = pd.Series(post_delta / 10, index=index, name="post")

    post_const = pd.Series(post_const, index=index, name="post_const")
    pre_const = pd.Series(pre_const, index=index, name="pre_const")

    if need_gap_constant:
        data = pd.concat(
            [pre, post, post_const, pre_const, gap_constant, gap_linear], axis=1
        )
    else:
        data = pd.concat([pre, post, post_const, pre_const], axis=1)

    return data


def load_qbo(pca=3):
    """
    Loads the QBO from https://acd-ext.gsfc.nasa.gov/Data_services/met/qbo/QBO_Singapore_Uvals_GSFC.txt'.
    If pca is set to an integer (default 3) then that many principal components are taken.
    If pca is set to 0 then the raw QBO data is returned.

    Parameters
    ----------
    pca : int, optional.  Default 3.
    """
    import sklearn.decomposition as decomp

    data = pd.read_table(
        "https://acd-ext.gsfc.nasa.gov/Data_services/met/qbo/QBO_Singapore_Uvals_GSFC.txt",
        skiprows=9,
        header=None,
        index_col=False,
        names=[
            "Month",
            "Year",
            "300",
            "250",
            "200",
            "150",
            "100",
            "90",
            "80",
            "70",
            "50",
            "40",
            "30",
            "20",
            "15",
            "10",
        ],
        usecols=range(16),
        sep=r"\s+",
    )
    data.index = pd.to_datetime(
        {"year": data["Year"], "month": data["Month"], "day": np.ones(len(data))}
    )
    data = data.drop(columns=["Year", "Month"])
    data.index = data.index.to_period(freq="M")

    if pca > 0:
        from string import ascii_lowercase

        pca_d = decomp.PCA(n_components=pca)
        for idx, c in zip(range(pca), ascii_lowercase, strict=False):
            data["pc" + c] = pca_d.fit_transform(data.values).T[idx, :]

    return data


def load_solar():
    """
    Gets the solar F10.7 from 'https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_all_years.dat'.
    """
    data = pd.read_table(
        "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_all_years.dat",
        header=None,
        sep=r"\s+",
    )
    data.index = pd.to_datetime(
        data[0] * 1000 + data[1], format="%Y%j"
    ) + pd.to_timedelta(data[2], unit="h")
    solar = pd.DataFrame(data[50]).rename(columns={50: "f10.7"})
    solar = solar.where(solar != 999.9)
    solar = solar.resample("MS").mean()
    return solar.to_period(freq="M")


def load_trop(deseasonalize=True):
    """
    Gets the tropical tropopause pressure from ftp.cdc.noaa.gov.  The tropical tropopause pressure is automatically
    deseasonalized by default to remove the strong seasonal cycle.

    Parameters
    ----------
    deseasonalize : bool, optional.  Default True
        If set to false deseasonalization will not be done.
    """
    path = "Datasets/ncep.reanalysis.derived/tropopause/"
    filename = "pres.tropp.mon.mean.nc"

    save_path = Path(appdirs.user_data_dir()) / filename
    directory = save_path.parent
    if not directory.exists():
        directory.mkdir(parents=True)

    # Only fetch from the ftp if the file does not exist or is greater than one week out of date.
    if (
        not save_path.exists()
        or time.time() - save_path.stat().st_mtime > 60 * 60 * 24 * 7
    ):
        ftp = ftplib.FTP("ftp.cdc.noaa.gov")
        ftp.login()
        ftp.cwd(path)
        ftp.retrbinary("RETR " + filename, Path.open(save_path, "wb").write)
        ftp.quit()

    data = xr.open_dataset(save_path)

    trop_only = (
        data.pres.mean(dim="lon")
        .where((data.lat > -5) & (data.lat < 5))
        .mean(dim="lat")
    )

    if deseasonalize:
        anom = trop_only.groupby("time.month") - trop_only.groupby("time.month").mean(
            dim="time"
        )
    else:
        anom = trop_only

    return anom.to_dataframe("pres").pres.to_period(freq="M")


def load_ao():
    """
    Loads the arctic oscillation index from ncep
    """
    data = pd.read_table(
        "http://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii",
        sep=r"\s+",
        header=None,
        names=["year", "month", "ao"],
    )

    data["dt"] = pd.to_datetime(
        {"year": data.year, "month": data.month, "day": np.ones(len(data))}
    ).dt.to_period(freq="M")

    return data.set_index(keys="dt")["ao"]


def load_aao():
    data = pd.read_table(
        "http://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/aao/monthly.aao.index.b79.current.ascii",
        sep=r"\s+",
        header=None,
        names=["year", "month", "aao"],
    )

    data["dt"] = pd.to_datetime(
        {"year": data.year, "month": data.month, "day": np.ones(len(data))}
    ).dt.to_period(freq="M")

    return data.set_index(keys="dt")["aao"]


def load_nao():
    """
    Loads the north atlantic oscillation index from noaa
    :return:
    """
    data = pd.read_table(
        "http://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii",
        sep=r"\s+",
        header=None,
        names=["year", "month", "nao"],
    )

    data["dt"] = pd.to_datetime(
        {"year": data.year, "month": data.month, "day": np.ones(len(data))}
    ).dt.to_period(freq="M")

    return data.set_index(keys="dt")["nao"]


def load_ehf(filename):
    """
    Loads the eddy heat flux data from the file erai_ehf_monthly_1978_2016.txt provided on the LOTUS ftp server in
    the folder Proxies-Weber
    """
    data = pd.read_table(
        filename,
        sep=r"\s+",
        header=None,
        skiprows=4,
        names=["year", "month", "sh_ehf", "nh_ehf"],
    )

    data["dt"] = pd.to_datetime(
        {"year": data.year, "month": data.month, "day": np.ones(len(data))}
    ).dt.to_period(freq="M")

    data = data.drop(["year", "month"], axis=1)

    return data.set_index(keys="dt")


def load_giss_aod():
    """
    Loads the giss aod index from giss
    """
    filename = "tau_reff_Sato-Lacis.nc"
    url = "https://data.giss.nasa.gov/modelforce/strataer/data/tau_reff_Sato-Lacis.nc"

    save_path = Path(appdirs.user_data_dir()) / filename
    _download_netcdf_if_needed(url, save_path)

    data = xr.open_dataset(save_path)

    data = data.mean(dim="lat")["tau"].to_dataframe()

    data.index = data.index.to_period(freq="M")
    data.index.names = ["time"]

    # Find the last valid non-zero entry and extend to the current date.
    valid_tau = data["tau"].dropna()
    last_nonzero_period = valid_tau[valid_tau != 0].index[-1]
    last_nonzero_idx = data.index.get_loc(last_nonzero_period)

    index = pd.period_range(data.index[0], end=_current_month_period(), freq="M")

    # New values
    vals = np.zeros(len(index))
    vals[: last_nonzero_idx + 1] = data["tau"].to_numpy()[: last_nonzero_idx + 1]
    vals[last_nonzero_idx + 1 :] = data["tau"].to_numpy()[last_nonzero_idx]

    return pd.Series(vals, index=index, name="aod")


def load_glossac_aod():
    filename = "tau_reff_GloSSAC.v2.24.nc"
    url = (
        "https://data.giss.nasa.gov/modelforce/strataer/data/"
        "tau_reff_GloSSAC.v2.24.nc"
    )

    save_path = Path(appdirs.user_data_dir()) / filename
    _download_netcdf_if_needed(url, save_path)

    data = xr.open_dataset(save_path)

    data = data.mean(dim="lat")["tau"].to_dataframe()
    data.index = data.index.to_period(freq="M")
    data.index.names = ["time"]

    index = pd.period_range(data.index[0], end=_current_month_period(), freq="M")
    aod = data["tau"].to_numpy()
    extended_aod = np.zeros(len(index))
    extended_aod[: len(aod)] = aod
    extended_aod[len(aod) :] = aod[-1]

    return pd.Series(extended_aod, index=index, name="aod")


def load_solar_mg2():
    """
    Loads the bremen solar composite mg2 index
    """
    data = pd.read_table(
        "http://www.iup.uni-bremen.de/gome/solar/MgII_composite.dat",
        sep=r"\s+",
        skiprows=23,
        names=["year", "month", "day", "index", "error", "id"],
        parse_dates={"time": [0, 1, 2]},
        index_col="time",
    )

    return data.resample("ME").mean().to_period(freq="M")["index"]


def load_orthogonal_eesc(filename):
    """
    Calculates two orthogonal eesc terms from the predicted eesc at 6 different ages of air, uses the EESC.txt
    datafile from the LOTUS ftp server in the folder EESC_Damadeo
    """
    data = pd.read_table(filename, sep=r"\s+", header=3)

    import sklearn.decomposition as decomp

    pca = decomp.PCA(n_components=2)
    data["eesc_1"], data["eesc_2"] = pca.fit_transform(data.values).T

    def frac_year_to_datetime(start):
        from datetime import datetime, timedelta

        year = int(start)
        rem = start - year

        base = datetime(year, 1, 1)
        return base + timedelta(
            seconds=(base.replace(year=base.year + 1) - base).total_seconds() * rem
        )

    data.index = data.index.map(frac_year_to_datetime)

    data = data.resample("MS").interpolate("linear")

    data.index = data.index.to_period(freq="M")

    data = data.drop(["1.0", "2.0", "3.0", "4.0", "5.0", "6.0"], axis=1)

    data /= data.std()

    return data


if __name__ == "__main__":
    load_solar()
