import datetime

import pandas as pd

# Functions that parse the raw data

def split_first_and_repeat_orders(data: pd.DataFrame, 
    user_col: str ='user_id_hash', 
    date_col: str ='order_created_at',
    order_col: str ='order_id_hash'):
    """ Split data into first and repeat orders
    Takes a Dataframe of orders and splits into a DataFrame of
    first time orders and a DataFrame of repeat orders.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset with orders, user identifiers, and date of order
    user_col : str
        Column name containing user identifier
    date_col : str
        Column name containing date of order (must be numeric or datetime)
    order_col : str
        Column name containing order identifier

    Returns
    -------
    2 Data Frames containing the First and Return Orders for 
    customers in the dataset.
    """

    first_order_time = (data[['user_id_hash', 'order_created_at']]
        .groupby('user_id_hash').min().reset_index())

    first_order_df = data.merge(first_order_time, how='inner', 
                          left_on=[user_col, date_col], 
                          right_on=[user_col, date_col])

    rep_orders_df = data[~data[order_col].isin(first_order_df[order_col])]

    return first_order_df, rep_orders_df

def split_to_holdout(data : pd.DataFrame, 
    split_prop : float=0.1,
    user_col: str ='user_id_hash', 
    date_col: str ='order_created_at',
    order_col: str ='order_id_hash'):
    """ Split holdout set from data, a percentage of repeat users (leaving first order in train)
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset with orders, user identifiers, and date of order
    split_prop : float
        Percent of return users to hold out
    user_col : str
        Column name containing user identifier
    date_col : str
        Column name containing date of order (must be numeric or datetime)
    order_col : str
        Column name containing order identifier

    Returns
    -------
    2 Data Frames containing the train and holdout sets for 
    customers in the dataset.
    """

    data['order_number'] = data.groupby(user_col)[date_col].rank(method='dense')

    multi_order_users = data\
        .loc[data['order_number']>1]['user_id_hash'].drop_duplicates()

    holdout_users = multi_order_users.sample(round(split_prop*len(multi_order_users)), 
        random_state=24601)

    holdout_set = data.loc[data['order_number']>1.0]\
        .loc[data[user_col].isin(holdout_users)]
    training_set = data.loc[~data[order_col]\
        .isin(holdout_set[order_col])]

    return training_set, holdout_set



