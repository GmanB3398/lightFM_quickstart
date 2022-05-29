import pandas as pd
import numpy as np

from lightfm.data import Dataset


def create_dataset(parsed_data: pd.DataFrame,
    user_col: str = 'user_id_hash', 
    item_col: str = 'catalog_item_id',
    user_attribute_cols: list = ['device'],
    item_attribute_cols: list = ['top_level_category', 'brand_name', 
        'subcategory', 'quantity'],
    user_identity_features: bool =True, 
    item_identity_features: bool =True):
    """ Create LightFM Dataset from user orders
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset with orders, user identifiers, and date of order
    user_col : str
        Column name containing user identifier
    item_col : str
        Column name containing order identifier
    user_attribute_cols : list
        Column name containing user identifier, pass empty list if not using
    item_attribute_cols : list
        Column name containing order identifier, pass empty list if not using
    user_identity_features: bool
        Add User ID Features to Dataset?
    item_identity_features: bool
        Add Item ID Features to Dataset?

    Returns
    -------
    A lightfm.Dataset with user-item combinations and attributes 
    """

    # Initialize Dataset
    dataset = Dataset(user_identity_features=user_identity_features, 
        item_identity_features=item_identity_features)
    # TODO: Test efficacy of setting these values to true and false

    # Give Dataset Basic Users and Items
    dataset.fit(users = parsed_data[user_col],
           items = parsed_data[item_col])

    # Add Item Attributes
    if len(item_attribute_cols) > 0:
        dataset.fit_partial(items = parsed_data[item_col],
            item_features = parsed_data[item_attribute_cols])

    # Add User Attributes
    if len(user_attribute_cols) > 0:
        dataset.fit_partial(items = parsed_data[user_col],
            item_features = parsed_data[user_attribute_cols])

    return dataset


def update_dataset(lfmDataset: Dataset,
    new_data: pd.DataFrame,
    user_col: str = 'user_id_hash', 
    item_col: str = 'catalog_item_id',
    user_attribute_cols: list = ['device'],
    item_attribute_cols: list = ['top_level_category', 'brand_name', 
        'subcategory', 'quantity']
        ):
    """ Updates LightFM Dataset with added data
    
    Parameters
    ----------
    lfmDataset : Dataset
        An Object of class Dataset from LightFM.data, fit on user_item pairs
    data : pd.DataFrame
        Dataset with orders, user identifiers, and date of order to be added
    user_col : str
        Column name containing user identifier
    item_col : str
        Column name containing order identifier
    user_attribute_cols : list
        Column name containing user identifier, pass empty list if not using
    item_attribute_cols : list
        Column name containing order identifier, pass empty list if not using

    Returns
    -------
    A lightfm.Dataset with user-item combinations and attributes 
    """

    # Give Dataset Basic Users and Items
    lfmDataset.fit_partial(users = new_data[user_col],
           items = new_data[item_col])

    # Add Item Attributes
    if len(item_attribute_cols) > 0:
        lfmDataset.fit_partial(items = new_data[item_col],
            item_features = new_data[item_attribute_cols])

    # Add User Attributes
    if len(user_attribute_cols) > 0:
        lfmDataset.fit_partial(items = new_data[user_col],
            item_features = new_data[user_attribute_cols])

    return lfmDataset

def get_lightfm_objs(lfmDataset: Dataset,
    data: pd.DataFrame,
    user_col: str = 'user_id_hash', 
    item_col: str = 'catalog_item_id',
    user_attribute_cols: list = ['device'],
    item_attribute_cols: list = ['top_level_category', 'brand_name', 
        'subcategory', 'quantity']):
    """ Get LightFM objects needed for training
    
    Parameters
    ----------
    lfmDataset : Dataset
        An Object of class Dataset from LightFM.data, fit on user_item pairs
    data : pd.DataFrame
        Dataset with orders, user identifiers, and date of order
    user_col : str
        Column name containing user identifier
    item_col : str
        Column name containing order identifier
    user_attribute_cols : list
        Column name containing user identifier, pass empty list if not using
    item_attribute_cols : list
        Column name containing order identifier, pass empty list if not using
    user_identity_features: bool
        Add User ID Features to Dataset?
    item_identity_features: bool
        Add Item ID Features to Dataset?

    Returns
    -------
    Sparse Matrix of interactions, Sparse Matrix of Item Features, Sparse Matrix of User Features 
    """

    (interactions, weights) = lfmDataset.build_interactions(list(zip(data[user_col],
        data[item_col])))

    if len(item_attribute_cols) > 0:
        item_features = lfmDataset.build_item_features(list(zip(data[item_col],
            [data[item_attribute_cols]])))
    else:
        item_features = None

    if len(user_attribute_cols) > 0:
        user_features = lfmDataset.build_item_features(list(zip(data[user_col],
            [data[user_attribute_cols]])))
    else:
        user_features = None

    return interactions, item_features, user_features
