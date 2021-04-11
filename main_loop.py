


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

    a = trading_algorithm.trader(d.df_live)
    a.get_account_info()


if __name__ == '__main__':

    main()

    # response = file_helper.get_data(
    #     data_prep.get_all_data,
    #     'prepared data.pkl',
    #     get_new = True
    # )

    # df_all_data, df_cb_cg_coins = response


    # make_model.make_model(df_all_data, df_cb_cg_coins)