import pandas as pd
import sklearn.tree as tree
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os



def save_K_fold(X, y, dirname, data_dir = '.'):
    skf = StratifiedKFold(n_splits = 10, shuffle=True, random_state=12345)
    skf.get_n_splits(X,y)

    taget_dir = os.path.join(data_dir,dirname) 
    if not os.path.exists(taget_dir):
        os.mkdir(taget_dir)
    
    i = 0
    for _, test_index in skf.split(X,y):
        target_file = 'cv' + str(i) + '.csv'
        new_X = X[test_index]
        new_y = y[test_index]
        new_Xy = np.c_[new_X,new_y]
        pd.DataFrame(new_Xy).to_csv(os.path.join(taget_dir,target_file),index=False)
        i += 1


def read_iris():
    dataframe = pd.read_csv("./data/original/iris/iris.data", names = ['speal_len','speal_width','petal_len','petal_width','class']) 
    print(dataframe.shape)
    data = dataframe.values
    delete_indexes = []
    for index, d in enumerate(data):
        if d[-1] == 'Iris-virginica':
            delete_indexes.append(index)
        elif d[-1] == 'Iris-setosa':
            data[index][-1] = '0'
        else:
            data[index][-1] = '1.0'

    data = np.delete(data, delete_indexes,axis=0)
    X = data[:, :-1]
    y = data[:,-1]
    save_K_fold(X,y, 'iris','./data')


def read_mushroom():
    dataframe = pd.read_csv("./data/original/mushroom/agaricus-lepiota.data",header=0) 
    print(dataframe.shape)
    data = dataframe.values
    delete_indexes = []
    for index, d in enumerate(data):
        if d[0] == 'p':
            data[index][0] = '0'
        else:
            data[index][0] = '1.0'

    data = np.delete(data, delete_indexes,axis=0)
    X = data[:, 1:]
    
    y = data[:, 0]
    save_K_fold(X,y, 'mushroom','./data')


def read_mushroom_k():
    # names = ['label', 'f1', 'f2', 'f3',]
    # dataframe = pd.read_csv("./data/original/mushroom/agaricus-lepiota.data",names=) 
    print(dataframe.shape)
    data = dataframe.values
    delete_indexes = []
    for index, d in enumerate(data):
        if d[0] == 'p':
            data[index][0] = '0'
        else:
            data[index][0] = '1.0'

    data = np.delete(data, delete_indexes,axis=0)
    X = data[:, 1:]
    
    y = data[:, 0]
    

read_mushroom()
# def read_car():
#     pass

# def read_adult():
#     pass