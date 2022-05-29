# Model Training Helpers
from typing import Optional

from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score, recall_at_k
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import scipy.sparse

from lightFM_quickstart.constants import hyper_param_grid_default


def best_param_finder(train_set: scipy.sparse.coo.coo_matrix,
                      val_set: scipy.sparse.coo.coo_matrix,
                      metric: str = 'precision',
                      k: Optional[int] = 10,
                      user_features: Optional[scipy.sparse.csr.csr_matrix] = None,
                      item_features: Optional[scipy.sparse.csr.csr_matrix] = None,
                      hyper_param_grid: list = hyper_param_grid_default):
    """ Find Parameters that maximize metric given on validation set
    
    Parameters
    ----------
    train_set : scipy.sparse.coo.coo_matrix
        Set of Interactions for Training
    val_set : scipy.sparse.coo.coo_matrix
        Set of Interactions for Testing Hyperparameters
    metric : str
        Either 'precision', 'recall', or 'auc' based on which metric to optimize for
    k : int
        For Precision and Recall, how many top ranked items to check.
    user_features : scipy.sparse.csr.csr_matrix
        User Features to be used by the model
    item_features: scipy.sparse.csr.csr_matrix
        Item features to be used by the model
    hyper_param_grid: list
        Hyperparameters to test, if none passed, default will be passed.


    Returns
    -------
    A dictionary of metric maximizing hyperparameters, optimal number of epochs
    """

    best_metric = 0
    best_epochs = 1
    best_params = {}

    print('Starting Hyperparameter optimization: {} Possibilities'
          .format(len(ParameterGrid(hyper_param_grid))))

    for params in tqdm(ParameterGrid(hyper_param_grid)):

        epochs = params.pop('epochs')
        trial_model = LightFM(loss='warp', **params)
        trial_model = trial_model.fit(train_set, epochs=epochs,
                                      item_features=item_features,
                                      user_features=user_features)

        if metric == 'precision':
            out_metric = precision_at_k(trial_model, val_set,
                                        train_interactions=train_set, k=k,
                                        item_features=item_features,
                                        user_features=user_features,
                                        check_intersections=False).mean()

        elif metric == 'recall':
            out_metric = recall_at_k(trial_model, val_set,
                                     train_interactions=train_set, k=k,
                                     item_features=item_features,
                                     user_features=user_features,
                                     check_intersections=False).mean()

        elif metric == 'auc':
            out_metric = auc_score(trial_model, val_set,
                                   train_interactions=train_set,
                                   item_features=item_features,
                                   user_features=user_features,
                                   check_intersections=False).mean()
        else:
            raise ValueError('Metric must be in: ["precision", "recall", "auc"]')

        if out_metric > best_metric:
            best_metric = out_metric
            best_params = params
            best_epochs = epochs

    print('Finished Hyperparameter optimization!')
    print(f'Optimal Hyperparameters: {best_params}')
    print(f'Optimal Epochs: {best_epochs}')

    return best_params, best_epochs
