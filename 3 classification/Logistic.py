import numpy as np
import file_io
from pre_work import *
from sklearn.linear_model import LogisticRegression

class Logistic:
    def __init__(self):
        pass


    def fit(self,x,y,lmd = 0.04,echo=2000):
        X = np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
        Y = np.zeros([y.shape[0],10])

        for i in range(len(y)):
            Y[i][int(y[i])] = 1

        self.W = np.random.randint(-10,10,size=[X.shape[1],10])
        self.W = np.array(self.W, dtype=float)

        for cnt in range(echo):
            g = 1.0/(1+np.exp(-1*X.dot(self.W)))
            self.W = self.W + lmd*(X.T.dot(Y-g))

    
    def predict(self,x):
        X = np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
        y = X.dot(self.W)
        Y = np.zeros(y.shape[0],dtype=int)

        for i in range(len(y)):
            idx = 0
            for j, num in enumerate(y[i]):
                if y[i][idx] < num:
                    idx = j
            Y[i] = idx

        return Y


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = file_io.mnist()

    logistic = Logistic()
    logistic.fit(x_train, y_train)
    y_predict = logistic.predict(x_test)

    sk_logistic = LogisticRegression()
    sk_logistic.fit(x_train,y_train)
    y_predict_sk = sk_logistic.predict(x_test)
    
    tol = len(y_test)

    score = 0
    sk_score = 0
    for i in range(tol):
        if y_test[i] == y_predict[i]:
            score += 1
        if y_test[i] == y_predict_sk[i]:
            sk_score += 1

    
    print("logistic precision:{}%\n".format(float(score)/tol*100))
    print("sk_logistic precision:{}%\n".format(float(sk_score)/tol*100))