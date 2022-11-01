from pre_work import *
import numpy as np
import matplotlib.pyplot as plt
import copy

class Linear:
    def __init__(self):
        pass

    def fit(self, x, y):
        m = x.shape[0]
        self.x = np.concatenate((np.ones((m,1)),x),axis=1)
        self.y = copy.copy(y)
        self.w = np.linalg.inv(np.transpose(self.x).dot(self.x)).dot(np.transpose(self.x)).dot(self.y)

    def predict(self, x):
        m = x.shape[0]
        X = np.concatenate((np.ones((m,1)),x),axis=1)
        y = X.dot(self.w)
        return y
        
if __name__ == '__main__':
    x_train, y_train = load_trainset()
    x_test, y_test = load_testset()

    linear = Linear()
    linear.fit(x_train, y_train)
    y_predict = linear.predict(x_test)

    plt.figure(figsize=(12, 6))                      
    plt.plot(y_test,label='True')
    plt.plot(y_predict,label='Line')

    plt.legend()
    plt.show()