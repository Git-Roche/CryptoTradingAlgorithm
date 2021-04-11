# import file_helper
import raw_data_process

import os

from pickle import load, dump

class dataprocessor():

    def __init__(self, new_data = False) -> None:
        self.df = None
        self.features = None
        self.target = None
        self.df_live = None

        self.new_data = new_data
        self.data_file = 'processed data.pkl'

    def get_data(self):

        if os.path.exists(self.data_file) and not self.new_data:
            print('loading existing prepared data')
            with open(self.data_file, 'rb') as f:
                data_dict = load(f)
                self.df = data_dict['df']
                self.features = data_dict['features']
                self.target = data_dict['target']
                self.df_live = data_dict['live']
        else:
            self.preprocess_data()

            print('saving data')
            data_dict = {}
            with open(self.data_file, 'wb') as f:
                data_dict['df'] = self.df
                data_dict['features'] = self.features
                data_dict['target'] = self.target
                data_dict['live'] = self.df_live
                dump(data_dict, f)

        

    def preprocess_data(self):

            
        raw_data_process.download_todays_data()
        self.df = raw_data_process.concat_existing_data()

        print('preparing data')

        print(self.df)

        # df.set_index(['dt', 'symbol'], inplace=True)

        # del df['datetime']

        self.df['relative_mcap'] = self.df.groupby('datetime')['market_caps'].transform(
            lambda x: x / x.sum()
        )

        # print(df.sort_values('datetime').head(50))

        self.features = ['prices', 'market_caps', 'total_volumes', 'relative_mcap']
        # df.set_index(['symbol', 'datetime'])

        time_horizons = [
            # 24 * 180,
            24 * 90,
            24 * 30, 
            24 * 7,
            24,
            8,
            4
        ]

        new_features = []

        self.df.set_index(['datetime'], inplace = True)
        #moving averages:
        for f in self.features:

            for t in time_horizons:
                #moving average
                feature_name = '{} MA {}'.format(f,t)
                new_features.append(feature_name)
                self.df[feature_name] = self.df.groupby('symbol')[f].transform(
                    lambda x: x.rolling(window = t).mean()
                    )
                #make relative to current value
                self.df[feature_name] = self.df.apply(axis = 1, func = lambda r: r[feature_name] / r[f])
                    
                #moving standard deviation
                feature_name = '{} MSTD {}'.format(f,t)
                new_features.append(feature_name)
                self.df[feature_name] = self.df.groupby('symbol')[f].transform(
                    lambda x: x.rolling(window = t).std()
                    )
                #make relative to current value
                self.df[feature_name] = self.df.apply(axis = 1, func = lambda r: r[feature_name] / r[f])

                #moving max
                feature_name = '{} MMAX {}'.format(f,t)
                new_features.append(feature_name)
                self.df[feature_name] = self.df.groupby('symbol')[f].transform(
                    lambda x: x.rolling(window = t).max()
                    )
                #make relative to current value
                self.df[feature_name] = self.df.apply(axis = 1, func = lambda r: r[feature_name] / r[f])

                #moving min
                feature_name = '{} MMIN {}'.format(f,t)
                new_features.append(feature_name)
                self.df[feature_name] = self.df.groupby('symbol')[f].transform(
                    lambda x: x.rolling(window = t).min()
                    )
                #make relative to current value
                self.df[feature_name] = self.df.apply(axis = 1, func = lambda r: r[feature_name] / r[f])


        #target
        self.target = 'prices +1H'
        self.df[self.target] = self.df.groupby('symbol')['prices'].transform(
            lambda x: x.shift(-1) / x
        )

        #make relative to prior values        
        for f in self.features:
            self.df[f] = self.df.groupby('symbol')[f].transform(
                lambda x: x.shift(1) / x
            )
            

        self.features += new_features


        #live data:
        max_date = self.df.reset_index()['datetime'].max()

        self.df_live = self.df.loc[max_date].copy()


        self.df.dropna(inplace=True)

        print(self.df[self.df['symbol']=='BTC'])