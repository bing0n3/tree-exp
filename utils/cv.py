import os
import numpy as np
import pandas as pd


def get_fold(path_name, k_fold):
    folds = []

    for i in range(10):
        f_name = os.path.join(path_name,"cv"+str(k_fold)+".csv")
        folds.append(pd.read_csv(f_name).values)

    test_X = folds[k_fold][:,:-1]
    test_y = folds[k_fold][:,-1]

    train_X = np.array([])
    train_y = np.array([])

    for i in range(10):
        if i == k_fold: 
            continue
        if train_X.size == 0 or train_y.size == 0:
            train_X = folds[i][:,:-1]
            train_y = folds[i][:,-1]
            continue
        train_X = np.vstack((train_X, folds[i][:,:-1]))
        # print(folds[i][:,:-1].shape,train_X.shape)
        train_y = np.append(train_y, folds[i][:,-1])


    return train_X, train_y, test_X, test_y


