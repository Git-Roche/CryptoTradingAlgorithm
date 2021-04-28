


# import data_prep

# import make_model

# import file_helper

import data_for_model
import model_class

import trading_algorithm

def main():


    d = data_for_model.dataprocessor(False)
    d.get_data()

    m = model_class.model(d.df, d.features, d.target)
    m.try_model()

    # d.get_recent_data()

    a = trading_algorithm.trader(m.df_prediction)
    a.get_account_info()
    a.get_order_book()
    a.get_acct_orders()
    # a.get_acct_fills()
    a.plan_orders()

if __name__ == '__main__':

    main()
