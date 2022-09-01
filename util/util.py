import platform
import random
import math
from typing import List ,Dict, Tuple

import pandas as pd
import numpy as np

import sklearn 
from sklearn.model_selection import StratifiedKFold , KFold
from sklearn.metrics import f1_score 
DATA_PATH = 'JobCare_data/'


def thresholds_search(y_true,y_prob):
  thresholds = np.linspace(0,1,101)
  f1_score_arr = np.array([f1_score(y_true,(y_prob>t).astype(np.int)) for t in thresholds])
  best_score = np.max(f1_score_arr)
  best_th = thresholds[np.argmax(f1_score_arr)]

  return best_score,best_th


def merge_codes(df:pd.DataFrame,df_code:pd.DataFrame,col:str)->pd.DataFrame:
    df = df.copy()
    df_code = df_code.copy()
    df_code = df_code.add_prefix(f"{col}_")
    df_code.columns.values[0] = col
    return pd.merge(df,df_code,how="left",on=col)


def preprocess_data(
                    df:pd.DataFrame,
                    is_train:bool = True,
                    cols_merge:List[Tuple[str,pd.DataFrame]] = []  ,
                    cols_equi:List[Tuple[str,str]]= [] ,
                    cols_drop:List[str] = []
                    )->Tuple[pd.DataFrame,np.ndarray]:
    df = df.copy()

    y_data = None
    if is_train:
        y_data = df["target"].to_numpy()
        df = df.drop(columns="target")

    for col, df_code in cols_merge:
        df = merge_codes(df,df_code,col)

    cols = df.select_dtypes(bool).columns.tolist()
    df[cols] = df[cols].astype(int)

    for col1, col2 in cols_equi:
        df[f"{col1}_{col2}"] = (df[col1] == df[col2] ).astype(int)

    df = df.drop(columns=cols_drop)
    return (df , y_data)

def cat_feature(x_train:pd.DataFrame):
    return x_train.columns[x_train.nunique() > 2].tolist()

def DataLoader():
    train_data = pd.read_csv(f'{DATA_PATH}train.csv')
    test_data = pd.read_csv(f'{DATA_PATH}test.csv')
    return train_data , test_data

def get_code():
    code_d = pd.read_csv(f'{DATA_PATH}속성_D_코드.csv').iloc[:,:-1]
    code_h = pd.read_csv(f'{DATA_PATH}속성_H_코드.csv')
    code_l = pd.read_csv(f'{DATA_PATH}속성_L_코드.csv')
    code_d.columns= ["attribute_d","attribute_d_d","attribute_d_s","attribute_d_m",'_']
    code_h.columns= ["attribute_h","attribute_h_m"]
    code_l.columns= ["attribute_l","attribute_l_d","attribute_l_s","attribute_l_m","attribute_l_l"]
    return code_d, code_h, code_l


train_data, test_data = DataLoader()
code_d, code_h, code_l = get_code()

cols_drop = ["id","person_prefer_f","person_prefer_g",
            "person_prefer_d_3_attribute_d_m_contents_attribute_d_attribute_d_m"]  
best_params = {"iterations": 1422,
            "objective": "CrossEntropy",
            "bootstrap_type": "Bayesian",
            "od_wait": 666,
            "learning_rate": 0.9782109291187356,
            "reg_lambda": 70.72533306533951,
            "random_strength": 47.81900485462368,
            "depth": 3,
            "min_data_in_leaf": 20,
            "leaf_estimation_iterations": 5,
            "one_hot_max_size": 1,
            "bagging_temperature": 0.07799233624102353,
            "eval_metric":"F1"}
cols_merge = [
        ("person_prefer_d_1" , code_d),
        ("person_prefer_d_2" , code_d),
        ("person_prefer_d_3" , code_d),
        ("contents_attribute_d" , code_d),
        ("person_prefer_h_1" , code_h),
        ("person_prefer_h_2" , code_h),
        ("person_prefer_h_3" , code_h),
        ("contents_attribute_h" , code_h),
        ("contents_attribute_l" , code_l)]
cols_equi = [

    ("contents_attribute_c","person_prefer_c"),
    ("contents_attribute_e","person_prefer_e"),

    ("person_prefer_d_2_attribute_d_s" , "contents_attribute_d_attribute_d_s"),
    ("person_prefer_d_2_attribute_d_m" , "contents_attribute_d_attribute_d_m"),
    ("person_prefer_d_2_attribute_d_d" , "contents_attribute_d_attribute_d_d"),
    ("person_prefer_d_3_attribute_d_s" , "contents_attribute_d_attribute_d_s"),
    ("person_prefer_d_3_attribute_d_m" , "contents_attribute_d_attribute_d_m"),
    ("person_prefer_d_3_attribute_d_d" , "contents_attribute_d_attribute_d_d"),

    ("person_prefer_h_1_attribute_h_m" , "contents_attribute_h_attribute_h_m"),

    ("person_prefer_h_2_attribute_h_m" , "contents_attribute_h_attribute_h_m"),
    ("person_prefer_h_3_attribute_h_m" , "contents_attribute_h_attribute_h_m")
]

def readyData():
    x_train, y_train = preprocess_data(train_data, cols_merge = cols_merge , cols_equi= cols_equi , cols_drop = cols_drop)
    x_test, _ = preprocess_data(test_data,is_train = False, cols_merge = cols_merge , cols_equi= cols_equi  , cols_drop = cols_drop)
    return x_train,y_train,x_test