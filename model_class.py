

# from sklearn.preprocessing import Normalizer
from normalizer_class import normalizer

from sklearn.feature_selection import f_regression

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

from sklearn import linear_model

import numpy as np



class model:

    def __init__(self, df, features, target) -> None:
        self.df = df
        self.features = features
        self.target = target

        self.df_prediction = None

        self.model_file = 'model.pkl'

    def try_model(self):
        print('making model')

        self.df.reset_index(inplace=True)
        self.df.set_index(['symbol', 'datetime'], inplace = True)

        x = self.df[self.features]
        y = self.df[self.target]

        x_train, x_test, y_train, y_test = train_test_split(x, y)

        # print(x_train)

        # n = normalizer()
        # n.fit(x_train)

        norm_x = normalizer()

        x_train = norm_x.fit_transform(x_train)
        x_test = norm_x.transform(x_test)
        norm_y = normalizer()

        y_train = norm_y.fit_transform(y_train)
        y_test = norm_y.transform(y_test)

        m = HistGradientBoostingRegressor(max_depth=5)
        # m = linear_model.LinearRegression()

        print('fitting')
        m.fit(x_train, np.ravel(y_train))

        print('getting scores')
        train_score = m.score(x_train, y_train)

        test_score = m.score(x_test, y_test)


        print('train score: {}'.format(train_score))
        print('test score: {}'.format(test_score))

        pred = m.predict(x_test)

        x_test['prediction'] = pred
        x_test['target'] = y_test

        def pct_err(r, a, b):
            return (r[a] - r[b]) / (r[b])

        x_test['pct_err'] = x_test.apply(axis = 1, func = pct_err, a = 'prediction', b='target')

        print(x_test.groupby('symbol')['pct_err'].mean())

        self.df.reset_index(inplace=True)
        self.df_prediction = self.df[self.df['datetime'] == self.df['datetime'].max()]
        
        x_live = self.df_prediction[self.features]
        x_live = norm_x.transform(x_live)

        self.df_prediction['prediction'] = m.predict(x_live)

        self.df_prediction = self.df_prediction[['symbol', 'datetime', 'prediction']]





    def fit_predict(self, df_live):
        """
        Fit the model on all the data then predict on live data.
        """
        self.df_live = df_live

        x = self.df[self.features]
        y = self.df[self.target]

        m = HistGradientBoostingRegressor(max_depth=5)

        norm_x = normalizer()

        x = norm_x.fit_transform(x)
        norm_y = normalizer()

        y = norm_y.fit_transform(y)

        m.fit(x,y)

        x_live = self.df_live[self.features]
        x_live = normalizer.transform(x_live)
        
        pred = m.predict(x_live)

        print('live prediction:')

        self.df_live['prediction'] = pred
        
        print(self.df_live['prediction'])

