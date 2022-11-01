from pre_work import *
import numpy as np
import matplotlib.pyplot as plt
import copy

class Ridge:
    def __init__(self):
        pass

    def fit(self, x, y, lmd = 0.2):
        m = x.shape[0]
        self.x = np.concatenate((np.ones((m,1)),x),axis=1)
        self.y = copy.copy(y)
        idty =  np.identity(len(self.x[0]))
        self.w = np.linalg.inv(np.transpose(self.x).dot(self.x)-lmd*idty).dot(np.transpose(self.x)).dot(self.y)

    def predict(self, x):
        m = x.shape[0]
        X = np.concatenate((np.ones((m,1)),x),axis=1)
        y = X.dot(self.w)
        return y

if __name__ == '__main__':
    x_train, y_train = load_trainset()
    x_test, y_test = load_testset()

    plt.figure(figsize=(12, 6)) 
    plt.plot(y_test,label='True')

    ridge = Ridge()

    ridge.fit(x_train, y_train,lmd=0)
    y_predict = ridge.predict(x_test)
    plt.plot(y_predict,label='Line lmd=0')

    ridge.fit(x_train, y_train,lmd=0.2)
    y_predict = ridge.predict(x_test)
    plt.plot(y_predict,label='Line lmd=0.2')

    ridge.fit(x_train, y_train,lmd=0.5)
    y_predict = ridge.predict(x_test)
    plt.plot(y_predict,label='Line lmd=0.5')

    plt.legend()
    plt.show()