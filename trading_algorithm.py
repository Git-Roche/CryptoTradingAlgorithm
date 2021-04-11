

import cbpro
# from pycoingecko import CoinGeckoAPI
import file_helper
from os.path import join
import api_keys

import pandas as pd

class trader:
    """
    Use the predicted future values to reallocate positions.
    """

    def __init__(self, df_pred) -> None:
        
        self.df_pred = df_pred

        #maximum fraction of total $ to put in a single coin
        self.max_share = 0.2

        self.folder = 'account_data'

        self.new_data = True
        #need list of tickers

        #need  current price

        #need order book distribution / std

        #need current orders

        #need current portfolio

    def get_account_info(self):

        # cbw = cbpro.WebsocketClient()
        
        cb = cbpro.AuthenticatedClient(api_keys.coinbase_api_key, api_keys.coinbase_api_secret, api_keys.coinbase_api_pass)

        
        # accounts = file_helper.get_data(cb.get_coinbase_accounts, join(self.folder,'accounts.pkl'), self.new_data)
        # # print(accounts)
        # df_acct = file_helper.dict_list_to_df(accounts)
        # print(df_acct)

        
        accounts = file_helper.get_data(cb.get_accounts, join(self.folder,'accounts.pkl'), self.new_data)

        df_acct = file_helper.dict_list_to_df(accounts)
        print(df_acct)

        df_acct = df_acct[df_acct['trading_enabled']==True]

        # accounts = file_helper.get_data(cb.get_orders, join(self.folder,'orders.pkl'), self.new_data)
        

        # def get_holds(account_id):
        #     response = cb.get_account_holds(account_id=account_id)
        #     holds = [r for r in response]
        #     return holds

        # account_holds = []
        # for a in accounts:
        #     response = file_helper.get_data(get_holds, join(self.folder,'{}.pkl'.format(a['name'])), self.new_data, account_id = a['id'])
        #     account_holds.append(response)

        # print(account_holds)

        def get_orders(status):
            response = cb.get_orders(status)
            orders = [r for r in response]
            return orders

        orders = file_helper.get_data(get_orders, join(self.folder,'orders.pkl'), self.new_data, status='all')
        
        df_orders = file_helper.dict_list_to_df(orders)
        print(df_orders)


        def get_fills():
            response = cb.get_fills()
            fills = [r for r in response]
            return fills

        fills = file_helper.get_data(get_fills, join(self.folder,'fills.pkl'), self.new_data)
        print(fills)
        # order_book = file_helper.get_data(cb.get_product_order_book, join(self.folder,'book.pkl'), self.new_data)
        # print(order_book)
        # account_history = file_helper.get_data(cb.get_account_history, join(self.folder,'history.pkl'), self.new_data)
        # print(account_history)





