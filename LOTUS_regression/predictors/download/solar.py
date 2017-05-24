import requests
import pandas as pd
import re


def load_solar():
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


if __name__ == "__main__":
    print(load_solar())
