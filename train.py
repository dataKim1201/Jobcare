import platform
import random
import math
from typing import List ,Dict, Tuple

import pandas as pd
import numpy as np
# import catboost
# import optuna
import sklearn 
from sklearn.model_selection import StratifiedKFold , KFold
from sklearn.metrics import f1_score 

from sklearn.model_selection import train_test_split

from util.util import *
from catboost import Pool,CatBoostClassifier
import model
SEED = 45
is_holdout = False
n_splits = 5

x_train,y_train,x_test = readyData()
cat_features = cat_feature(x_train)

model_s, score = model.model.catboo(x_train,y_train,cat_features)
print(score)