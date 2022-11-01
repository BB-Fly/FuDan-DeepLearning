import numpy as np
import math
import copy

def nor(X, axis=-1, p=2):
    x = X
    lp_norm = np.atleast_1d(np.linalg.norm(x, p, axis))
    lp_norm[lp_norm == 0] = 1
    return (x / np.expand_dims(lp_norm, axis))

def sig(X):
    x = 1/(1+pow(math.e,-X))
    return x

def std(X):
    x = copy.deepcopy(X)
    for i in range (x.shape[0]):
        r = max(x[i])
        l = min(x[i])
        for j in range(x.shape[1]):
            x[i][j] = (x[i][j]-l)/r-l
    return x

def squ(X):
    x = copy.deepcopy(X)
    for i in x:
        for j in i:
            j = j**3

    return x

def mx(X):
    x = copy.deepcopy(X)
    for i in range (x.shape[0]):
        r = max(x[i])
        l = min(x[i])
        for j in range(x.shape[1]):
            x[i][j] = x[i][j]-(r-l)/2
    return x

def gs(X):
    x = squ(std(X))
    return pow(math.e,-x)
