
from math import floor
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

from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# from sklearn import tree

import cbpro


def make_model(df_all_data, df_cb_cg_coins):
    
    print('running model')

    # symbol_list = []
    # test_r2 = []
    # train_r2 = []

    df_cb_cg_coins.sort_values('base_currency', inplace=True)

    df_nona = df_all_data.dropna().copy()

    x_columns = [
            '30 Day Price Mean Pct',
            '7 Day Price Mean Pct',
            '1 Day Price Mean Pct',
            '4 Hour Price Mean Pct',
            '-1 Hour Price Pct',
        
            '30 Day Volume Mean Pct',
            '7 Day Volume Mean Pct',
            '1 Day Volume Mean Pct',
            '4 Hour Volume Mean Pct',
            '-1 Hour Volume Pct'
        ]

    

    def get_pred(r, model):
        x_vals = []
        for x in x_columns:
            x_vals.append(r[x])

        return model.predict([x_vals])[0]

    # for symbol in df_cb_cg_coins['base_currency']:
    # df = df_nona[df_nona['symbol'] == symbol]


    # if len(df.index) < 10:
    #     print('{} - not enough data'.format(symbol))
    #     continue

    df_reg = df_nona[[

        *x_columns,
        '+1 Hour Price Up',
        '+1 Hour Price Pct',
        'prices'
    ]].copy()

    # df_reg.dropna(inplace=True)


    # random_filter = np.random.rand(len(df_reg)) < 0.8


    x_data = df_reg[x_columns]

    # x_train = x_data[random_filter]
    # x_test = x_data[~random_filter]

    y_data = df_reg[['+1 Hour Price Up']]

    # y_train = y_data[random_filter]
    # y_test = y_data[~random_filter]

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size = 0.33,
        shuffle = True
    )

    # df_reg = df_reg.loc[:]

    # model = DecisionTreeRegressor(max_depth=5)
    model = RandomForestRegressor(max_depth=10)
    # model = MLPClassifier(
    #     hidden_layer_sizes=(100,100),
    #     activation='logistic',
    #     learning_rate='adaptive',
    #     max_iter=10000
    # )
    # model = MLPRegressor(
    #     hidden_layer_sizes=(200,200,200,100,100,100),
    #     activation='tanh',
    #     learning_rate='adaptive',
    #     max_iter=10000
    # )
    model.fit(
        X = x_train,
        y = y_train.values.ravel()
        )

    # plot_tree(regressor, max_depth=5)

    # print()



    #apply trading strategy to measure profit
    df_nona['prediction'] = df_nona.apply(axis = 1, 
        func = get_pred,
        model = model
    )
    # model.predict(df_reg[x_columns])



    # print('\n{} regression results'.format(symbol))
    # y_pred = model.predict(x_test)
    # print(classification_report(y_test, y_pred))

    # print('train:')
    # print(model.score(x_train, y_train))
    # print('test:')
    # print(model.score(x_test, y_test))

    # symbol_list.append(symbol)
    # train_r2.append(model.score(x_train, y_train))
    # test_r2.append(model.score(x_test, y_test))

    print(model.score(x_train, y_train))
    print(model.score(x_test, y_test))

    # plt.show()

    # results_data = {
    #     'symbol' : symbol_list,
    #     'train_r2' : train_r2,
    #     'test_r2' : test_r2
    # }


    # df_results = pd.DataFrame(data = results_data)


    # print('regression results')
    # print(df_results.sort_values('test_r2', ascending=False))


    df_nona.reset_index(inplace=True)

    #round to the nearest hour
    time_round = 3600000.0
    df_nona['rounded_timestamp'] = df_nona.apply(
        axis=1,
        func = lambda r: int(floor(r['timestamp'] / time_round) * time_round)
        )
    

    def add_one_hour(r):
        """
        coingecko doesn't return time stamps at regular intervals.
        sometimes they are two close to the same hour in a row.
        for testing purposes, re apply the ordering of the time for each symbol
        """
        if len(r) == 1:
            return r.iloc[0]
        else:
            return int(r.iloc[0] + time_round) #datetime.timedelta(hours=1)


    df_nona['rounded_timestamp_fixed'] = df_nona.groupby('symbol')['rounded_timestamp'].transform(
        lambda d: d.rolling(2, min_periods = 1).apply(add_one_hour)
    )
    # min_datetime = df_nona['rounded_datetime'].min()

    df_nona['rounded_datetime'] = df_nona.apply(
        axis = 1, 
        func = lambda r: datetime.datetime.utcfromtimestamp(r['rounded_timestamp_fixed'] / 1000.0)
        )


    # print(df_nona[df_nona['rounded_timestamp_fixed'] != df_nona['rounded_timestamp']])

    # print(df_nona)

    # def get_funds(r):
    #     print(r)
    #     return 1

    # df_nona['funds'] = df_nona.groupby('rounded_datetime')['prices', 'prediction'].transform(
    #     lambda d: d.rolling(2, min_periods = 1).apply(get_funds)
    # )

    # df_nona['funds'] = 0.0


    print(df_nona[['rounded_datetime', 'symbol', 'prices', '+1 Hour Price Pct']])

    df_nona.sort_values('rounded_datetime', inplace = True)

    fund_list = []

    starting_funds = 10000
    df_temp_prior = pd.DataFrame()

    for t in df_nona['rounded_datetime'].drop_duplicates():

        df_temp = df_nona[(df_nona['rounded_datetime'] == t) & (df_nona['prediction'] >= 0)].copy()
        
        total = df_temp['prediction'].sum()

        if int(total) == 0 and not df_temp_prior.empty:
            df_temp['funds'] = df_temp_prior['funds'] * (1.0 + df_temp_prior['+1 Hour Price Pct'])
        elif int(total) == 0:
            continue
        elif df_temp_prior.empty:
            #distribute starting funds proportional to normalized prediction
            # df_nona['funds'].mask(df_nona['rounded_datetime'] == t, )
            df_temp['funds'] = df_temp.apply(axis = 1, func = lambda r: r['prediction'] / total * starting_funds)
        
        else:
            #get new total funds with price changes
            # df_temp_prior = df_nona[df_nona['rounded_datetime'] == t_prior]

            current_funds = (df_temp_prior['funds'] * (1.0 - df_temp_prior['+1 Hour Price Pct'])).sum()

            #redistribute
            df_temp['funds'] = df_temp.apply(axis = 1, func = lambda r: r['prediction'] / total * current_funds)

        # t_prior = t
        df_temp_prior = df_temp

        fund_list.append(df_temp[['rounded_datetime', 'symbol', 'prediction', 'funds']].copy())

    # print(df_nona[['rounded_datetime', 'prices', 'prediction', 'funds']])

    df_funds = pd.concat(fund_list)

    df_grouped = df_funds.groupby('rounded_datetime')[['funds']].sum()

    print(df_grouped)

    fig, ax = plt.subplots(figsize=(12,8))

    df_grouped.plot(ax = ax)

    plt.tight_layout()
    plt.show()
    # plt.clo

    # df_nona['total_buys'] = df_nona.groupby('rounded_datetime')['prediction'].transform('sum')

    #how to redistribute funds
    # df_nona['normalized_prediction'] = df_nona['prediction'] / df_nona['total_buys']



    # df_pivot = df_nona.pivot_table(
    #     index = 'rounded_datetime',
    #     columns = 'symbol',
    #     values = 'prediction'
    #     )

    # starting_funds = 10000
    # def get_funds(r):
        
    #     if len(r.index) == 1:
    #         return starting_funds
        
    #     #weighted sum of all of them
    #     total = 0
    #     for s in r.iloc[0]:
    #         if not pd.isna(s):
    #             total += s



    # df_pivot['funds'] = df_pivot.rolling(2, min_periods = 1).apply(
    #     func = get_funds
    # )

    # print(df_pivot)
    # def get_holdings(r):



    # df_nona['holdings'] = df_nona.rolling(2, min_periods = 1).apply(
    #     func = 
    # )


    # print(df_nona)
    #calculate profits



