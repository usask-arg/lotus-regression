import ftplib
import xarray as xr
import os
import appdirs
import time


def load_trop():
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

    anom = trop_only.groupby('time.month') - trop_only.groupby('time.month').mean(dim='time')

    return anom.to_dataframe('pres').pres.to_period(freq='M')

if __name__ == "__main__":
    print(load_trop())
