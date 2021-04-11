
from os import path
from pickle import load, dump
import pandas as pd
from time import sleep

def get_data(function, fpath, get_new = False,  **kwargs):

    """
    get data from api as dictionary using function
    kwargs get sent to function
    save result to fpath

    if result already exists at fpath, then load fpath pickle data instead
    """
        
    if not path.exists(fpath) or get_new:
        print('getting new {}'.format(fpath))
        response = function(**kwargs)

        #stay below 3 requests per second limit on coinbase pro
        sleep(0.5)

        with open(fpath, 'wb') as f:
            dump(response, f)
    else:
        print('loading existing {}'.format(fpath))
        
        with open(fpath, 'rb') as f:
            response = load(f)

    # df = pd.DataFrame.from_dict(response)

    return response

# def get_df(fpath):
#     if not path.exists(fpath):
#         print('downloading new {}'.format(fpath))
#         response = function(**kwargs)

#         #stay below 3 requests per second limit on coinbase pro
#         sleep(0.5)

#         with open(fpath, 'wb') as f:
#             dump(response, f)
#     else:
#         print('loading existing {}'.format(fpath))
        
#         with open(fpath, 'rb') as f:
#             response = load(f)

#     df = pd.DataFrame.from_dict(response)

#     return df


def dict_list_to_df(dict_list):
    
        keys = []
        for a in dict_list:
            for k in a:
                if not k in keys:
                    keys.append(k)
        account_dict = {}
        for k in keys:
            account_dict[k] = []
        for a in dict_list:
            for k in keys:
                if k in a:
                    account_dict[k].append(a[k])
                else:
                    account_dict[k].append(None)

        df = pd.DataFrame(account_dict)

        return df