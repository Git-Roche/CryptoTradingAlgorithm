

import cbpro
# from pycoingecko import CoinGeckoAPI
import file_helper
from os.path import join
import api_keys
from copy import copy
import pandas as pd

class trader:
    """
    Use the predicted future values to reallocate positions.
    """

    def __init__(self, df_pred) -> None:
        
        self.df_pred = df_pred

        self.cb = cbpro.AuthenticatedClient(api_keys.coinbase_api_key, api_keys.coinbase_api_secret, api_keys.coinbase_api_pass)

        #maximum fraction of total $ to put in a single coin
        self.max_share = 0.2

        self.folder = 'account_data'

        self.new_data = False
        #need list of tickers
        self.df_acct = None

        #need  current price

        #need order book distribution / std
        self.df_order_book = None

        #need current orders
        self.df_orders = None

        #need current portfolio

    def get_order_book(self):

        self.df_acct['product'] = self.df_acct['currency'].apply(lambda x: str(x) + '-USD')

        print(self.df_acct)

        order_book_list = []
        for p, c in zip(self.df_acct['product'], self.df_acct['currency']):
            fpath = join('order book', '{}.pkl'.format(p))
            order_book = file_helper.get_data(self.cb.get_product_order_book, fpath, self.new_data, product_id=p, level=2)
            if 'message' in order_book:
                print('{} not found in order books.'.format(p))
                continue
            df_book = pd.DataFrame(order_book)
            df_book['product'] = p
            df_book['currency'] = c
            order_book_list.append(df_book.copy())
            # cb.get_product_order_book(product_id=p, level=2)

        df_order_books = pd.concat(order_book_list)

        df_order_books['bid_price'] = df_order_books['bids'].apply(lambda x: x[0])
        df_order_books['bid_size'] = df_order_books['bids'].apply(lambda x: x[1])
        df_order_books['bid_num_orders'] = df_order_books['bids'].apply(lambda x: x[2])
        df_order_books['ask_price'] = df_order_books['asks'].apply(lambda x: x[0])
        df_order_books['ask_size'] = df_order_books['asks'].apply(lambda x: x[1])
        df_order_books['ask_num_orders'] = df_order_books['asks'].apply(lambda x: x[2])
        
        print(df_order_books)

        df_min_book = df_order_books.groupby('currency').agg(
            {
                'bid_price' : 'max',
                'ask_price' : 'min'
            }
        )
        df_min_book.reset_index(inplace=True)
        self.df_order_book = df_min_book
        print(self.df_order_book)


    def get_account_info(self):

        # cbw = cbpro.WebsocketClient()
        

        
        # accounts = file_helper.get_data(cb.get_coinbase_accounts, join(self.folder,'accounts.pkl'), self.new_data)
        # # print(accounts)
        # df_acct = file_helper.dict_list_to_df(accounts)
        # print(df_acct)

        
        accounts = file_helper.get_data(self.cb.get_accounts, join(self.folder,'accounts.pkl'), self.new_data)

        self.df_acct = file_helper.dict_list_to_df(accounts)
        print(self.df_acct)

        self.df_acct = self.df_acct[self.df_acct['trading_enabled']==True]

        # accounts = file_helper.get_data(cb.get_orders, join(self.folder,'orders.pkl'), self.new_data)
        
        self.df_acct = self.df_acct.merge(
            self.df_pred, 
            left_on = 'currency', 
            right_on='symbol', 
            suffixes=('', '_r'), 
            how='left'
        )

        def get_target_pct(r):
            if r['prediction'] < 0.0 or pd.isna(r['prediction']):
                return 0.0
            else: 
                return r['prediction']
            
            

        self.df_acct['target_pct'] = self.df_acct.apply(axis = 1, func = get_target_pct)
        self.df_acct['target_pct'] = self.df_acct['target_pct'] / self.df_acct['target_pct'].sum()

        print(self.df_acct)

        # def get_holds(account_id):
        #     response = cb.get_account_holds(account_id=account_id)
        #     holds = [r for r in response]
        #     return holds

        # account_holds = []
        # for a in accounts:
        #     response = file_helper.get_data(get_holds, join(self.folder,'{}.pkl'.format(a['name'])), self.new_data, account_id = a['id'])
        #     account_holds.append(response)

        # print(account_holds)

    def get_acct_orders(self):

        # self.cb.get_orders

        def get_orders(status):

            response = self.cb.get_orders(status)
            orders = []
            for r in response:
                if 'message' not in r:
                    orders.append(r)
            # orders = [r for r in response]
            return orders

        orders = file_helper.get_data(get_orders, join(self.folder,'orders.pkl'), self.new_data, status='all')
        
        if orders:
            self.df_orders = file_helper.dict_list_to_df(orders)
        else:
            self.df_orders = pd.DataFrame()
        print(self.df_orders)

    def get_acct_fills(self):
        def get_fills():
            response = self.cb.get_fills()
            fills = []
            for r in response:
                if 'message' not in r:
                    fills.append(r)
            # fills = [r for r in response]
            return fills

        fills = file_helper.get_data(get_fills, join(self.folder,'fills.pkl'), self.new_data)
        
        if fills:
            self.df_fills = file_helper.dict_list_to_df(fills)
        else:
            self.df_fills = pd.DataFrame()
        print(self.df_fills)

        # order_book = file_helper.get_data(cb.get_product_order_book, join(self.folder,'book.pkl'), self.new_data)
        # print(order_book)
        # account_history = file_helper.get_data(cb.get_account_history, join(self.folder,'history.pkl'), self.new_data)
        # print(account_history)



    def plan_orders(self):

        self.df_acct = self.df_acct.merge(self.df_order_book, left_on='currency', right_on='currency', suffixes=('','_ob'), how='left')

        if not self.df_orders.empty:
            self.df_acct = self.df_acct.merge(self.df_orders, left_on='currency', right_on='currency', suffixes=('','_o'), how='left')

        
        # 

        # self.df_acct.dropna(inplace=True)

        self.df_acct['dollar_ballance'] = self.df_acct.apply(axis = 1, func = lambda x: float(x['balance']) * float(x['ask_price']))

        print(self.df_acct)
        cash = float(self.df_acct[self.df_acct['currency'] == 'USD']['balance'].iloc[0])
        total_bal = self.df_acct['dollar_ballance'].sum() + cash

        self.df_acct['target_value'] = self.df_acct['target_pct'] * total_bal
        self.df_acct['target_bal'] = self.df_acct.apply(axis = 1, func = lambda r: float(r['target_value']) * float(r['bid_price']))

        self.df_acct['bal_delta'] = self.df_acct.apply(axis = 1, func = lambda r: float(r['balance']) - float(r['target_bal']))
        
        usd_per_btc = float(self.df_acct[self.df_acct['currency'] == 'BTC']['bid_price'].iloc[0])
        
        self.df_acct['target_value_btc'] = self.df_acct.apply(
            axis=1, 
            func=lambda r: r['target_value'] / usd_per_btc
        )
        self.df_acct['current_value_btc'] = self.df_acct.apply(
            axis = 1, 
            func = lambda r: r['dollar_ballance'] / usd_per_btc
        )
        self.df_acct['delta_value_btc'] = self.df_acct['current_value_btc'] - self.df_acct['target_value_btc']

        df_bal = self.df_acct[self.df_acct['bal_delta'] != 0]
        df_bal.dropna(inplace=True)


        total_surplus_btc = df_bal[df_bal['delta_value_btc'] < 0]['delta_value_btc'].sum()
        total_demand_btc = df_bal[df_bal['delta_value_btc'] > 0]['delta_value_btc'].sum()

        print(df_bal)

        print('BTC demand: {}, BTC surplus {}'.format(total_demand_btc, total_surplus_btc))


        trades_to_do = {
            'currency_from' : [],
            'currency_to' : [],
            'amount_from' : [],
            'amount_to' : []
        }

        df_shit_coins = df_bal[df_bal['currency'] != 'BTC'].copy()

        df_shit_coins['amount_to'] = abs(df_shit_coins['bal_delta'])
        df_shit_coins['amount_from'] = abs(df_shit_coins['delta_value_btc'])

        #trade from usd to currency

        #trade from currencty to btc

        #trade from btc to currency

        trades_to_do['currency_to'] = df_shit_coins['currency'].to_list()
        trades_to_do['currency_from'] = ['BTC' for _ in df_shit_coins.index]
        trades_to_do['amount_to'] = df_shit_coins['amount_to'].to_list()
        trades_to_do['amount_from'] = df_shit_coins['amount_from'].to_list()

        # print(trades_to_do)

        df_trades = pd.DataFrame(trades_to_do)

        print(df_trades)
        # print(total_bal)

        #plan USD buys
        
        #plan trades to BTC

        #plan trades from BTC

        #take current balance and do trades to get to target_bal

