# Recommender Template
## By Griffin Barich

This Project contains tools to get a LightFM recommender system up and running very quickly from a source dataset.

## Dataset Creation and Training

Dataset creation is done using the example set up in the LightFM documentation. We fit a sparse matrix dataset with user-item interactions, adding columns and rows for these item and user features. This module then uses a cross-validated splitting function provided by LightFM to split the data randomly into a Training (60%), Validation (20%), and Testing (20%) Set. We also split into a first visit set and a return visit set.

We then run a hyperparameter optimization using a Grid Search based on the constants list in `constants.py`. These hyperparameters are used to find the model that optimizes the training set on the validation set for the precision at K metric (default: k=10). We then report the precision of the model trained on the training and validation set on the test set and train a model on the full data. You can pass in your own hyper-parameter grid as well.

### How to Run

To run the Training pipeline, pip install the `lightFM_quickstart` module 

```{python3}
from lightFM_quickstart.quickstart import lightfmModel, train_model_random_split
import pandas as pd

df = pd.read_csv('data/example_data.csv.gz')

hyper_param_grid = [{
    "no_components": [6, 12],
    "learning_schedule": ["adagrad", "adadelta"],
    "max_sampled": [5, 15],
    "item_alpha": [0, 1e-6],
    "epochs": [1, 10]
}]

model = train_model_random_split(df, 'user_id', 'order_id', 'item_id', 'order_date',
                                 user_attribute_cols=['device'], k=10,
                                 hyper_param_grid=hyper_param_grid)

results = model.predict_user(user_id)
```
Training takes a long time, so be sure to cache trained model in pickle files or similar.

### Known Tech Debt

- Add Split by time
- Add Evaluation functions

## References 

.gitignore from *equinor*'s [Data Science Template](https://github.com/equinor/data-science-template)

