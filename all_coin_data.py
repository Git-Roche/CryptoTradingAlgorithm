
import api_keys
from pycoingecko import CoinGeckoAPI
import file_helper
from pickle import dump, load

from os import path
import os

import pandas as pd
import datetime

import numpy as np
import matplotlib.ticker as mtick

from matplotlib import pyplot as plt
# import dateutil

from sklearn.neural_network import MLPClassifier

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
# from sklearn import tree

import cbpro

today = datetime.datetime.now().date()

folder_str = '{}{}{} coin data'.format(today.year, today.month, today.day)

if not path.exists(folder_str):
    os.mkdir(folder_str)


supported_coins_file = 'supported coins.pkl'

cg = CoinGeckoAPI()
#get supported coins

df_cg_coins = file_helper.get_data(cg.get_coins_list, path.join(folder_str, supported_coins_file))

# df_cg_coins.to_excel(path.join(folder_str, 'coingecko products.xlsx'))


print('coin gecko supported coins:')
print(df_cg_coins)

#get cg markets per coin
cg_coin_markets_file = 'coingecko coin markets.pkl'
df_cg_coin_markets = file_helper.get_data(cg.get_coins_markets, path.join(folder_str, cg_coin_markets_file), vs_currency='USD')
print('coingecko coin markets:')
print(df_cg_coin_markets)
# df_cg_coin_markets.to_excel(path.join(folder_str, 'coingecko coin markets.xlsx'))

cb = cbpro.AuthenticatedClient(api_keys.coinbase_api_key, api_keys.coinbase_api_secret, api_keys.coinbase_api_pass)

coinbase_products_file = 'coinbase products.pkl'

df_cb_coins = file_helper.get_data(cb.get_products, path.join(folder_str, coinbase_products_file))

print('coinbase supported coins:')
print(df_cb_coins)




df_cb_usd = df_cb_coins[df_cb_coins['quote_currency'] == 'USD']
df_cg_coin_markets['symbol'] = df_cg_coin_markets.apply(axis=1, func= lambda r: str.upper(r['symbol']))

#df_cg_coins
#df_cg_coin_markets
df_cb_cg_coins = df_cb_usd.merge(
    df_cg_coin_markets, 
    left_on='base_currency', 
    right_on='symbol',
    suffixes=('_cb', '_cg'),
    validate='1:1'
    )
# df_cb_cg_coins.to_excel(path.join(folder_str, 'matched products.xlsx'))
# print(df_cb_cg_coins)

# df.to_excel(path.join(folder_str, 'coinbase products.xlsx'))
# with open(path.join(folder_str, coinbase_products_file), 'rb') as f:
#     response = load(f)

# print(response)

#pick the top mcap for 2 same symbols.
# df_cb_cg_coins['test_currency'] = 'test'
df_cb_cg_coins['symbol_mcap_rank'] = df_cb_cg_coins.groupby('base_currency').transform('rank',  ascending=False)['market_cap']
df_cb_cg_coins = df_cb_cg_coins[df_cb_cg_coins['symbol_mcap_rank'] == 1]
del df_cb_cg_coins['symbol_mcap_rank']

# print(df_cb_cg_coins)

#get list of all 90 day coin data that is both in coin gecko and coinbase pro
coin_data_list = []
for cg_coin, symbol in zip(df_cb_cg_coins['id_cg'], df_cb_cg_coins['base_currency']):
    fpath = path.join(folder_str, '{} - 90 day data.pkl'.format(symbol))

    df = file_helper.get_data(
        function = cg.get_coin_market_chart_by_id,
        fpath = fpath,
        id = cg_coin,
        vs_currency = 'USD',
        days = 90
        )

    df['symbol'] = symbol

    coin_data_list.append(df)

print('concat dataframes')
df_all_data = pd.concat(coin_data_list)


df_all_data['datetime'] = df_all_data.apply(
    axis = 1, 
    func = lambda r: datetime.datetime.utcfromtimestamp(r['prices'][0] / 1000.0))
    
df_all_data['timestamp'] = df_all_data.apply(axis = 1, func = lambda r: r['prices'][0])

df_all_data['timestamp_min'] = df_all_data.groupby('symbol').transform('min')['timestamp']
df_all_data['timestamp_index'] = df_all_data['timestamp'] - df_all_data['timestamp_min']

df_all_data['prices'] = df_all_data.apply(axis = 1, func = lambda r: r['prices'][1])
df_all_data['market_caps'] = df_all_data.apply(axis = 1, func = lambda r: r['market_caps'][1])
df_all_data['total_volumes'] = df_all_data.apply(axis = 1, func = lambda r: r['total_volumes'][1])

df_all_data.set_index(df_all_data['datetime'], inplace=True)
del df_all_data['datetime']


    # cg.get_coin_market_chart_by_id(cg_coin, 'USD', 90)



df_all_data['30 Day Price Mean'] = df_all_data.groupby('symbol')['prices'].transform(
    lambda x: x.rolling(window = 24 * 30).mean()
    )

df_all_data['7 Day Price Mean'] = df_all_data.groupby('symbol')['prices'].transform(
    lambda x: x.rolling(window = 24 * 7).mean()
    )

df_all_data['1 Day Price Mean'] = df_all_data.groupby('symbol')['prices'].transform(
    lambda x: x.rolling(window = 24).mean()
    )
    
df_all_data['4 Hour Price Mean'] = df_all_data.groupby('symbol')['prices'].transform(
    lambda x: x.rolling(window = 4).mean()
    )

def get_pct_error(r, c1, c2):
    """
    get percent error
    r = dataframe row
    c1 = column name 1
    c2 = column name 2
    """
    if r[c2] != 0:
        return (r[c1] - r[c2]) / r[c2]

def get_up_down(r, c1, c2):
    """
    return 1 if it went up, otherwise return 0
    """
    if r[c1] < r[c2]:
        return 1
    else:
        return 0


df_all_data['30 Day Price Mean Pct'] = df_all_data.apply(axis=1, func = get_pct_error, c1='prices', c2='30 Day Price Mean')
df_all_data['7 Day Price Mean Pct'] = df_all_data.apply(axis=1, func = get_pct_error, c1='prices', c2='7 Day Price Mean')
df_all_data['1 Day Price Mean Pct'] = df_all_data.apply(axis=1, func = get_pct_error, c1='prices', c2='1 Day Price Mean')
df_all_data['4 Hour Price Mean Pct'] = df_all_data.apply(axis=1, func = get_pct_error, c1='prices', c2='4 Hour Price Mean')

df_all_data['-1 Hour Price'] = df_all_data.groupby('symbol')['prices'].transform(
    lambda x: x.shift(1)
)

df_all_data['+1 Hour Price'] = df_all_data.groupby('symbol')['prices'].transform(
    lambda x: x.shift(-1)
)

df_all_data['+1 Hour Price Pct'] = df_all_data.apply(axis=1, func = get_pct_error, c1='prices', c2='+1 Hour Price')
df_all_data['-1 Hour Price Pct'] = df_all_data.apply(axis=1, func = get_pct_error, c1='prices', c2='-1 Hour Price')
df_all_data['+1 Hour Price Up'] = df_all_data.apply(axis=1, func = get_up_down, c1='prices', c2='+1 Hour Price')

print(df_all_data)

# df_all_data.to_excel(path.join(folder_str,'check all data.xlsx'))

print('running regression')

symbol_list = []
test_r2 = []
train_r2 = []

for symbol in df_cb_cg_coins['base_currency']:
    df = df_all_data[df_all_data['symbol'] == symbol]
    df_reg = df[[
        '30 Day Price Mean Pct',
        '7 Day Price Mean Pct',
        '1 Day Price Mean Pct',
        '4 Hour Price Mean Pct',
        '-1 Hour Price Pct',
        '+1 Hour Price Up'
        # '+1 Hour Price Pct'
    ]].copy()

    df_reg.dropna(inplace=True)

    if len(df_reg.index) < 10:
        print('{} - not enough data'.format(symbol))
        continue

    random_filter = np.random.rand(len(df_reg)) < 0.8

    x_data = df_reg[[
            '30 Day Price Mean Pct',
            '7 Day Price Mean Pct',
            '1 Day Price Mean Pct',
            '4 Hour Price Mean Pct',
            '-1 Hour Price Pct'
        ]]

    # x_train = x_data[random_filter]
    # x_test = x_data[~random_filter]

    y_data = df_reg[['+1 Hour Price Up']]

    # y_train = y_data[random_filter]
    # y_test = y_data[~random_filter]

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size = 0.2
    )

    # df_reg = df_reg.loc[:]

    # model = DecisionTreeRegressor(max_depth=5)
    # model = RandomForestRegressor(max_depth=15)
    model = MLPClassifier(
        hidden_layer_sizes=(100,100),
        activation='logistic',
        learning_rate='adaptive',
        max_iter=10000
    )
    model.fit(
        X = x_train,
        y = y_train.values.ravel()
        )

    # plot_tree(regressor, max_depth=5)

    # print('\n{} regression results'.format(symbol))
    # print('train:')
    # print(regressor.score(x_train, y_train))
    # print('test:')
    # print(regressor.score(x_test, y_test))

    symbol_list.append(symbol)
    train_r2.append(model.score(x_train, y_train))
    test_r2.append(model.score(x_test, y_test))

    # plt.show()

results_data = {
    'symbol' : symbol_list,
    'train_r2' : train_r2,
    'test_r2' : test_r2
}

df_results = pd.DataFrame(data = results_data)


print('regression results')
print(df_results.sort_values('test_r2', ascending=False))

#create plots for all symbols
pic_folder = '{}{}{} pics'.format(today.year, today.month, today.day)

if not path.exists(pic_folder):
    os.mkdir(pic_folder)

for symbol in df_cb_cg_coins['base_currency']:
    print('making plot for {}'.format(symbol))
    df = df_all_data[df_all_data['symbol'] == symbol]

    fig, ax1 = plt.subplots(figsize=(16,8))
    
    ax1.set_ylabel('{}-USD'.format(symbol))

    ax1.plot(df['prices'], label='Price')
    ax2 = ax1.twinx()

    ax2.plot(df['total_volumes'], color='r', label='Volume')
    ax2.set_ylabel('Volume')

    fmt = '${x:,.2f}'
    tick = mtick.StrMethodFormatter(fmt)
    ax1.yaxis.set_major_formatter(tick) 
    
    fmt = '{x:,.0f}'
    tick = mtick.StrMethodFormatter(fmt)
    ax2.yaxis.set_major_formatter(tick) 

    fig.legend()
    fig.tight_layout()

    fpath = path.join(pic_folder, '{} - 90 day plot.png'.format(symbol))
    plt.savefig(fpath)

    plt.close()
    # plt.show()





# print(df_ranked)

# if path.exists('response.pkl'):
#     print('loading existing data')
    
#     with open('response.pkl', 'rb') as f:
#         response = load(f)

# else:
#     print('downloading data from coingecko')
    

#     response = cg.get_coin_market_chart_by_id('bitcoin', 'USD', 90)

#     with open('response.pkl', 'wb') as f:
#         dump(response, f)

# # print(response)

# df = pd.DataFrame.from_dict(response)


# df['datetime'] = df.apply(
#     axis = 1, 
#     func = lambda r: datetime.datetime.utcfromtimestamp(r['prices'][0] / 1000.0))

# df['prices'] = df.apply(axis = 1, func = lambda r: r['prices'][1])
# df['market_caps'] = df.apply(axis = 1, func = lambda r: r['market_caps'][1])
# df['total_volumes'] = df.apply(axis = 1, func = lambda r: r['total_volumes'][1])

# # df['datetime'] = datetime.datetime.utcfromtimestamp() 

# df.set_index(df['datetime'], inplace=True)
# del df['datetime']

# print(df)
# fig, ax1 = plt.subplots()
# ax1.plot(df['prices'])

# ax2 = ax1.twinx()
# ax2.plot(df['total_volumes'], color='r')

# fig.tight_layout()

# plt.show()