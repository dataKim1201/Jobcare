import os
import sys
import platform
import random
import math
from typing import List ,Dict, Tuple

import pandas as pd
import numpy as np
import catboost

import sklearn 
from sklearn.model_selection import StratifiedKFold , KFold
from sklearn.metrics import f1_score 

from catboost import Pool,CatBoostClassifier
from util.util import best_params


def catboo(x_train,y_train,cat_features):
    SEED = 43
    is_holdout = False
    n_splits = 5
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    scores = []
    models = []


    for tri, vai in cv.split(x_train):
        model = CatBoostClassifier(**best_params,cat_features = cat_features)
        model.fit(x_train.iloc[tri], y_train[tri],
            eval_set=[(x_train.iloc[vai], y_train[vai])],
            verbose =True
        )
        models.append(model)
        scores.append(model.get_best_score()["validation"]["F1"])
        if is_holdout:
            break
    print(scores)
    print(np.mean(scores))
    return models, scores
