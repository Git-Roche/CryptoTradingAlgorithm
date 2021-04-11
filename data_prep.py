
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


def get_all_data():
    today = datetime.datetime.now().date()

    month_str = '0' + str(today.month)
    month_str = month_str[-2:]
    day_str = '0' + str(today.day)
    day_str = day_str[-2:]

    folder_str = '{}{}{} coin data'.format(today.year, month_str, day_str)

    if not path.exists(folder_str):
        os.mkdir(folder_str)


    supported_coins_file = 'supported coins.pkl'

    cg = CoinGeckoAPI()
    #get supported coins

    response = file_helper.get_data(cg.get_coins_list, path.join(folder_str, supported_coins_file))

    df_cg_coins = pd.DataFrame.from_dict(response)

    # df_cg_coins.to_excel(path.join(folder_str, 'coingecko products.xlsx'))


    print('coin gecko supported coins:')
    print(df_cg_coins)

    #get cg markets per coin
    cg_coin_markets_file = 'coingecko coin markets.pkl'

    response = file_helper.get_data(cg.get_coins_markets, path.join(folder_str, cg_coin_markets_file), vs_currency='USD')
    df_cg_coin_markets = pd.DataFrame.from_dict(response)
    print('coingecko coin markets:')
    print(df_cg_coin_markets)
    # df_cg_coin_markets.to_excel(path.join(folder_str, 'coingecko coin markets.xlsx'))

    cb = cbpro.AuthenticatedClient(api_keys.coinbase_api_key, api_keys.coinbase_api_secret, api_keys.coinbase_api_pass)

    coinbase_products_file = 'coinbase products.pkl'

    response = file_helper.get_data(cb.get_products, path.join(folder_str, coinbase_products_file))
    df_cb_coins = pd.DataFrame.from_dict(response)

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

    #Filter out stable coin
    # df_cb_cg_coins[df_cb_cg_coins['base_currency'] != 'DAI']

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

        response = file_helper.get_data(
            function = cg.get_coin_market_chart_by_id,
            fpath = fpath,
            id = cg_coin,
            vs_currency = 'USD',
            days = 90
            )

        df = pd.DataFrame.from_dict(response)

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


    #volume
    
    df_all_data['30 Day Volume Mean'] = df_all_data.groupby('symbol')['total_volumes'].transform(
        lambda x: x.rolling(window = 24 * 30).mean()
        )

    df_all_data['7 Day Volume Mean'] = df_all_data.groupby('symbol')['total_volumes'].transform(
        lambda x: x.rolling(window = 24 * 7).mean()
        )

    df_all_data['1 Day Volume Mean'] = df_all_data.groupby('symbol')['total_volumes'].transform(
        lambda x: x.rolling(window = 24).mean()
        )
        
    df_all_data['4 Hour Volume Mean'] = df_all_data.groupby('symbol')['total_volumes'].transform(
        lambda x: x.rolling(window = 4).mean()
        )
    
    df_all_data['30 Day Volume Mean Pct'] = df_all_data.apply(axis=1, func = get_pct_error, c1='total_volumes', c2='30 Day Volume Mean')
    df_all_data['7 Day Volume Mean Pct'] = df_all_data.apply(axis=1, func = get_pct_error, c1='total_volumes', c2='7 Day Volume Mean')
    df_all_data['1 Day Volume Mean Pct'] = df_all_data.apply(axis=1, func = get_pct_error, c1='total_volumes', c2='1 Day Volume Mean')
    df_all_data['4 Hour Volume Mean Pct'] = df_all_data.apply(axis=1, func = get_pct_error, c1='total_volumes', c2='4 Hour Volume Mean')

    df_all_data['-1 Hour Volume'] = df_all_data.groupby('symbol')['total_volumes'].transform(
        lambda x: x.shift(1)
    )

    df_all_data['+1 Hour Volume'] = df_all_data.groupby('symbol')['total_volumes'].transform(
        lambda x: x.shift(-1)
    )

    df_all_data['+1 Hour Volume Pct'] = df_all_data.apply(axis=1, func = get_pct_error, c2='total_volumes', c1='+1 Hour Volume')
    df_all_data['-1 Hour Volume Pct'] = df_all_data.apply(axis=1, func = get_pct_error, c1='total_volumes', c2='-1 Hour Volume')
    df_all_data['+1 Hour Volume Up'] = df_all_data.apply(axis=1, func = get_up_down, c1='total_volumes', c2='+1 Hour Volume')


    return [df_all_data, df_cb_cg_coins]