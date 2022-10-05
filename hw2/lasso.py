from pre_work import *
import numpy as np
import matplotlib.pyplot as plt
import copy

class Lasso:
    def __init__(self):
        pass

    def fit(self, x, y, lmd = 0.2, learning_rate = 0.000005, epochs = 50000):
        m = x.shape[0]
        self.x = np.concatenate((np.ones((m,1)),x),axis=1)
        self.y = copy.copy(y)

        xMAT = np.mat(self.x)
        yMAT = np.mat(self.y.reshape(-1,1))

        self.w = np.ones(self.x.shape[1]).reshape(-1,1)

        for i in range(epochs):
            gradient = xMAT.T * (xMAT*self.w - yMAT)/m + lmd*np.sign(self.w)
            self.w = self.w -  learning_rate*gradient


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

    lasso = Lasso()

    lasso.fit(x_train, y_train,lmd=0)
    y_predict = lasso.predict(x_test)
    plt.plot(y_predict,label='Line lmd=0')

    lasso.fit(x_train, y_train,lmd=0.2)
    y_predict = lasso.predict(x_test)
    plt.plot(y_predict,label='Line lmd=0.2')

    lasso.fit(x_train, y_train,lmd=0.5)
    y_predict = lasso.predict(x_test)
    plt.plot(y_predict,label='Line lmd=0.5')

    plt.legend()
    plt.show()