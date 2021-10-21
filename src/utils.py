import os
import copy
from os.path import join as pjoin
from collections import Counter
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import torch

from imblearn.over_sampling import RandomOverSampler

_dir_path = os.path.dirname(os.path.realpath(__file__))


def load_data():
    return pd.read_json(pjoin("../input/data.json"))


def create_test_set(data):
    data_test = data.loc[data.level.isnull()]
    data_train = data.loc[~data.level.isnull()]
    return data_train, data_test


def create_folds(X):
    X["kfold"] = -1
    X = X.sample(frac=1).reset_index(drop=True)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X=X, y=X.level.values)):
        X.loc[val_idx, 'kfold'] = fold

    return X


def random_over_sampling(X, y):
    ros = RandomOverSampler(random_state=42)

    X_ros, y_ros = ros.fit_resample(X, y)
    print(X_ros.shape[0] - X.shape[0], 'new random picked points')
    print(sorted(Counter(y_ros).items()))

    return X_ros, y_ros


def get_train_data(df, fold):
    print("Getting the dataset")
    FOLD_MAPPPING = {
        0: [1, 2, 3, 4],
        1: [0, 2, 3, 4],
        2: [0, 1, 3, 4],
        3: [0, 1, 2, 4],
        4: [0, 1, 2, 3]
    }
    df = copy.deepcopy(df)
    train_df = df[df.kfold.isin(FOLD_MAPPPING.get(fold))].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)
    y_train = train_df.level.values
    y_valid = valid_df.level.values
    X_train = train_df.drop(["level", "kfold"], axis=1)
    X_valid = valid_df.drop(["level", "kfold"], axis=1)
    return X_train, X_valid, y_train, y_valid


def save_model(model, optimizer, model_name):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, pjoin(_dir_path, 'models', f'{model_name}.pth'))