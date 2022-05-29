import datetime
from typing import Optional
import warnings

warnings.filterwarnings('ignore', '.*OpenMP.*')  # For the LightFM Warnings

import pandas as pd
from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import precision_at_k

from lightFM_quickstart.data_cleaning import split_first_and_repeat_orders, split_to_holdout
from lightFM_quickstart.lightfm_preprocessing import create_dataset, get_lightfm_objs
from lightFM_quickstart.model_training_helpers import best_param_finder
from lightFM_quickstart.constants import hyper_param_grid_default


class lightfmModel:
    def __init__(self,
                 df: pd.DataFrame,
                 user_id_col: str,
                 order_id_col: str,
                 item_id_col: str,
                 date_col: Optional[str] = None,
                 use_columns: Optional[list] = None,
                 user_attribute_cols: list = [],
                 item_attribute_cols: list = []):
        """
        Parameters
        ----------
        df: data of all user's orders and items
        user_id_col: column in df with user identifiers
        order_id_col: column in df with order identifiers
        item_id_col: column in df with item identifiers
        date_col: column in df with date identifiers (not needed if not using holdout set)
        use_columns: list of columns to use in analysis
        user_attribute_cols: columns that are user attributes
        item_attribute_cols: columns that are item attributes
        """
        self.df = df
        if use_columns is not None:
            self.df = self.df[use_columns]

        self.user_id_col = user_id_col
        self.item_id_col = item_id_col
        self.order_id_col = order_id_col
        self.date_col = date_col

        self.user_attribute_cols = user_attribute_cols
        self.item_attribute_cols = item_attribute_cols

        self.file_prefix = str(datetime.datetime.now())[0:10].replace('-', '')

    def create_holdout_set(self, split_prop):
        cleaned_data, holdout_data = split_to_holdout(self.df,
                                                      split_prop=split_prop,
                                                      user_col=self.user_id_col,
                                                      date_col=self.date_col,
                                                      order_col=self.order_id_col)
        self.df = cleaned_data
        self.holdout_df = holdout_data

    def split_df_repeats(self):
        first_order_df, rep_orders_df = split_first_and_repeat_orders(data=self.df,
                                                                      user_col=self.user_id_col,
                                                                      date_col=self.date_col,
                                                                      order_col=self.order_id_col)
        self.first_order_df = first_order_df
        self.rep_orders_df = rep_orders_df

    def create_lightfm_objects_random(self):

        self.lfm_dataset = create_dataset(self.df,
                                          user_col=self.user_id_col,
                                          item_col=self.item_id_col,
                                          user_attribute_cols=self.user_attribute_cols,
                                          item_attribute_cols=self.item_attribute_cols)

        self.interactions, self.item_features, self.user_features = get_lightfm_objs(self.lfm_dataset,
                                                                                     self.df,
                                                                                     user_col=self.user_id_col,
                                                                                     item_col=self.item_id_col,
                                                                                     user_attribute_cols=self.user_attribute_cols,
                                                                                     item_attribute_cols=self.item_attribute_cols)

        # Split Interactions into Training (60%), Validation (20%) and Testing (20%)
        (self.rand_train_val_interactions, self.rand_test_interactions) = random_train_test_split(self.interactions,
                                                                                             test_percentage=0.2,
                                                                                             random_state=24601)

        (self.rand_train_interactions, self.rand_val_interactions) = random_train_test_split(
            self.rand_train_val_interactions,
            test_percentage=0.25,
            random_state=24601)

    def create_lightfm_objects_repeats(self):
        self.lfm_dataset = create_dataset(self.df,
                                          user_col=self.user_id_col,
                                          item_col=self.item_id_col,
                                          user_attribute_cols=self.user_attribute_cols,
                                          item_attribute_cols=self.item_attribute_cols)

        self.train_interactions, self.train_item_features, self.train_user_features = get_lightfm_objs(self.lfm_dataset,
                                                                                                       self.first_order_df,
                                                                                                       user_col=self.user_id_col,
                                                                                                       item_col=self.item_id_col,
                                                                                                       user_attribute_cols=self.user_attribute_cols,
                                                                                                       item_attribute_cols=self.item_attribute_cols)

        self.test_interactions, self.test_item_features, self.test_user_features = get_lightfm_objs(self.lfm_dataset,
                                                                                                    self.rep_orders_df,
                                                                                                    user_col=self.user_id_col,
                                                                                                    item_col=self.item_id_col,
                                                                                                    user_attribute_cols=self.user_attribute_cols,
                                                                                                    item_attribute_cols=self.item_attribute_cols)

    def train_model_random(self,
                           k: int = 10,
                           hyper_param_grid: list = hyper_param_grid_default):
        best_params, epochs = best_param_finder(train_set=self.rand_train_interactions,
                                                val_set=self.rand_val_interactions,
                                                metric='precision', k=k,
                                                user_features=self.user_features,
                                                item_features=self.item_features,
                                                hyper_param_grid=hyper_param_grid)

        model = LightFM(loss='warp', **best_params)
        model = model.fit(self.rand_train_val_interactions, epochs=epochs,
                          item_features=self.item_features,
                          user_features=self.user_features)

        precision = precision_at_k(model, self.rand_test_interactions, train_interactions=None,
                                   user_features=self.user_features, k=k,
                                   item_features=self.item_features).mean()

        full_model = LightFM(loss='warp', **best_params)
        full_model = full_model.fit(self.interactions, epochs=epochs,
                                    item_features=self.item_features,
                                    user_features=self.user_features)
        self.full_model = full_model

    def create_mappings_for_prediction(self):
        self.user_mappings = self.lfm_dataset.mapping()[0]
        self.item_mappings = self.lfm_dataset.mapping()[2]

    def predict_user(self, user_id: str):
        user_prediction_df = pd.DataFrame(list(zip(list(self.item_mappings.keys()),
                                                   self.full_model.predict(user_ids=self.user_mappings[user_id],
                                                                           item_ids=list(self.item_mappings.values()),
                                                                           item_features=self.item_features,
                                                                           user_features=self.user_features))),
                                          columns=['item_id', 'preds']).sort_values('preds', ascending=False)

        user_prediction_df = user_prediction_df.rename(columns={'preds': 'score'}).dropna()

        user_prediction_dict = user_prediction_df.to_dict('index')

        return user_prediction_dict


def train_model_random_split(df: pd.DataFrame,
                             user_id_col: str,
                             order_id_col: str,
                             item_id_col: str,
                             date_col: Optional[str] = None,
                             use_columns: Optional[list] = None,
                             user_attribute_cols: list = [],
                             item_attribute_cols: list = [],
                             k: int = 10,
                             hyper_param_grid: list = hyper_param_grid_default) -> lightfmModel:
    """ Wrapper to Train LightFM Recommender using custom functions from cleaned dataset

        Parameters
        ----------
        df: data of all user's orders and items
        user_id_col: column in df with user identifiers
        order_id_col: column in df with order identifiers
        item_id_col: column in df with item identifiers
        date_col: column in df with date identifiers (not needed if not using holdout set)
        use_columns: list of columns to use in analysis
        user_attribute_cols: columns that are user attributes
        item_attribute_cols: columns that are item attributes
        k: k for precision at k metric
        hyper_param_grid: list of length 1 with dictionary of hyper parameters

        Returns
        -------
        object
        Object of Class lightfmModel
        """

    rec_model = lightfmModel(df,
                             user_id_col, order_id_col, item_id_col, date_col,
                             use_columns,
                             user_attribute_cols,
                             item_attribute_cols)

    rec_model.create_lightfm_objects_random()

    rec_model.train_model_random(k=k,
                                 hyper_param_grid=hyper_param_grid)

    rec_model.create_mappings_for_prediction()

    return rec_model
