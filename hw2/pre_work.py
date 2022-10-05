import numpy as np
from sklearn import datasets


def get_array(X,Y):
    X = np.array(X)
    Y = np.transpose(np.array(Y))
    return X,Y



def load_dataset():
    # 获取数据集
    # x表示房屋属性，共13项，y表示房价
    train_x, train_y = datasets.load_boston(return_X_y = True)
    return get_array(train_x,train_y)

def load_trainset():
    X, Y = load_dataset()
    return X[0:450], Y[0:450]

def load_testset():
    X, Y = load_dataset()
    return X[451:506], Y[451:506]