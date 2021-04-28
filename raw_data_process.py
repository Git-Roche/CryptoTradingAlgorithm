

from datetime import datetime
from pickle import load
import file_helper
from os import path
import os
from pycoingecko import CoinGeckoAPI
import pandas as pd
import api_keys
import cbpro
from datetime_helper import round_to_nearest_hour

from copy import copy, deepcopy
#want to get all the latest data into a single dataframe with only relavent info for the model

class data_downloader:

    def __init__(self, new_data=False) -> None:
        
        self.new_data = new_data
        self.df_cg_coins = None


    def download_todays_data(self):
        """
        download the lastest data from coingecko 

        """
        today = datetime.now().date()

        month_str = '0' + str(today.month)
        month_str = month_str[-2:]
        day_str = '0' + str(today.day)
        day_str = day_str[-2:]

        folder_str = '{}{}{} coin data'.format(today.year, month_str, day_str)

        if not path.exists(folder_str):
            os.mkdir(folder_str)
        else:
            print('"{}" already exists.\nSkipping downloads.'.format(folder_str))
            return


        supported_coins_file = 'supported coins.pkl'

        cg = CoinGeckoAPI()
        #get supported coins

        response = file_helper.get_data(cg.get_coins_list, path.join(folder_str, supported_coins_file))

        self.df_cg_coins = pd.DataFrame.from_dict(response)

        # self.df_cg_coins.to_excel(path.join(folder_str, 'coingecko products.xlsx'))


        print('coin gecko supported coins:')
        print(self.df_cg_coins)

        #get cg markets per coin
        cg_coin_markets_file = 'coingecko coin markets.pkl'

        response = file_helper.get_data(cg.get_coins_markets, path.join(folder_str, cg_coin_markets_file), vs_currency='USD')
        self.df_cg_coin_markets = pd.DataFrame.from_dict(response)
        print('coingecko coin markets:')
        print(self.df_cg_coin_markets)
        # self.df_cg_coin_markets.to_excel(path.join(folder_str, 'coingecko coin markets.xlsx'))

        cb = cbpro.AuthenticatedClient(api_keys.coinbase_api_key, api_keys.coinbase_api_secret, api_keys.coinbase_api_pass)

        coinbase_products_file = 'coinbase products.pkl'

        response = file_helper.get_data(cb.get_products, path.join(folder_str, coinbase_products_file))
        self.df_cb_coins = pd.DataFrame.from_dict(response)

        print('coinbase supported coins:')
        print(self.df_cb_coins)




        df_cb_usd = self.df_cb_coins[self.df_cb_coins['quote_currency'] == 'USD']
        self.df_cg_coin_markets['symbol'] = self.df_cg_coin_markets.apply(axis=1, func= lambda r: str.upper(r['symbol']))

        #self.df_cg_coins
        #self.df_cg_coin_markets
        df_cb_cg_coins = df_cb_usd.merge(
            self.df_cg_coin_markets, 
            left_on='base_currency', 
            right_on='symbol',
            suffixes=('_cb', '_cg'),
            validate='1:1'
            )

        #pick the top mcap for 2 same symbols.
        # df_cb_cg_coins['test_currency'] = 'test'
        df_cb_cg_coins['symbol_mcap_rank'] = df_cb_cg_coins.groupby('base_currency').transform('rank',  ascending=False)['market_cap']
        df_cb_cg_coins = df_cb_cg_coins[df_cb_cg_coins['symbol_mcap_rank'] == 1]
        del df_cb_cg_coins['symbol_mcap_rank']

        #get list of all 90 day coin data that is both in coin gecko and coinbase pro
        for cg_coin, symbol in zip(df_cb_cg_coins['id_cg'], df_cb_cg_coins['base_currency']):
            fpath = path.join(folder_str, '{} - 90 day data.pkl'.format(symbol))

            response = file_helper.get_data(
                function = cg.get_coin_market_chart_by_id,
                fpath = fpath,
                id = cg_coin,
                vs_currency = 'USD',
                days = 90
                )

    def get_1_day_data(self):
        pass

    def concat_existing_data(self) -> pd.DataFrame:
        """
        This will search all of the coin data folders and get the 90 day data into a single dataframe and return it.
        """

        #get the folders
        dir_info = os.listdir()

        df_list = []
        for d in dir_info:
            if os.path.isdir(d) and d.endswith('coin data'):
                for file in os.listdir(d):
                    fpath = path.join(d,file)
                    if path.isfile(fpath) and file.endswith('90 day data.pkl'):
                        with open(fpath, 'rb') as f:
                            dict = load(f)
                            df = pd.DataFrame.from_dict(dict)
                            symbol = file.split(' ')[0]
                            df['symbol'] = symbol
                            df_list.append(deepcopy(df))

        # if not df_list:
        
        print('merging dataframes')
        df_data = pd.concat(df_list)


        df_data['datetime'] = df_data.apply(
            axis = 1, 
            func = lambda r: datetime.utcfromtimestamp(r['prices'][0] / 1000.0)
            )
            
        # df_data['timestamp'] = df_data.apply(axis = 1, func = lambda r: r['prices'][0])

        # df_data['timestamp_min'] = df_data.groupby('symbol').transform('min')['timestamp']
        # df_data['timestamp_index'] = df_data['timestamp'] - df_data['timestamp_min']


        df_data['prices'] = df_data.apply(axis = 1, func = lambda r: r['prices'][1])
        df_data['market_caps'] = df_data.apply(axis = 1, func = lambda r: r['market_caps'][1])
        df_data['total_volumes'] = df_data.apply(axis = 1, func = lambda r: r['total_volumes'][1])
        df_data['datetime'] = df_data['datetime'].apply(round_to_nearest_hour)

        df_data.drop_duplicates(subset=['symbol', 'datetime'], inplace=True)

        print(df_data)



        return df_data
