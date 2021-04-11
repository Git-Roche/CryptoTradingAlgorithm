

from pycoingecko import CoinGeckoAPI

from pickle import dump, load

from os import path

import pandas as pd
import datetime

from matplotlib import pyplot as plt
# import dateutil

if path.exists('response.pkl'):
    print('loading existing data')
    
    with open('response.pkl', 'rb') as f:
        response = load(f)

else:
    print('downloading data from coingecko')
    cg = CoinGeckoAPI()

    response = cg.get_coin_market_chart_by_id('bitcoin', 'USD', 90)

    with open('response.pkl', 'wb') as f:
        dump(response, f)

# print(response)

df = pd.DataFrame.from_dict(response)

# print(df)
# print(datetime.datetime.utcoffset())
# print(datetime.datetime.utcnow())

# e = datetime.datetime.utcfromtimestamp(0)
# t = (datetime.datetime.now() - e).total_seconds() * 1000.0
# print(e)
# print(t)

# tz = dateutil.tz.gettz('America / Phoenix')

df['datetime'] = df.apply(
    axis = 1, 
    func = lambda r: datetime.datetime.utcfromtimestamp(r['prices'][0] / 1000.0))

df['prices'] = df.apply(axis = 1, func = lambda r: r['prices'][1])
df['market_caps'] = df.apply(axis = 1, func = lambda r: r['market_caps'][1])
df['total_volumes'] = df.apply(axis = 1, func = lambda r: r['total_volumes'][1])

# df['datetime'] = datetime.datetime.utcfromtimestamp() 

df.set_index(df['datetime'], inplace=True)
del df['datetime']

print(df)
fig, ax1 = plt.subplots()
ax1.plot(df['prices'])

ax2 = ax1.twinx()
ax2.plot(df['total_volumes'], color='r')

fig.tight_layout()

plt.show()
# print(datetime.datetime.utcfromtimestamp(1602908217))
# print(datetime.datetime.utcfromtimestamp(1610681489))
# print(datetime.datetime.(1602908217015))

