import numpy as np
import file_io


class Least_squares:
    def __init__(self):
        pass

    def fit(self, x, y):
        m = x.shape[0]
        X = np.concatenate((np.ones((m,1)),x),axis=1)
        Y = np.zeros([len(y),10])
        for i in range(len(y)):
            Y[i][int(y[i])] = 1
        
        tmp = X.T.dot(X)
        for i in range(len(tmp)):
            tmp[i][i] += 1e-9

        self.w = np.linalg.inv(tmp).dot(X.T).dot(Y)

    def predict(self, x):
        m = x.shape[0]
        X = np.concatenate((np.ones((m,1)),x),axis=1)
        y = X.dot(self.w)
        Y = np.zeros(len(x))
        for i in range(len(y)):
            idx=0
            for j in range(10):
                if y[i][j] > y[i][idx]:
                    idx = j
            Y[i] = idx
        return Y

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = file_io.mnist()

    least_squares = Least_squares()
    least_squares.fit(x_train, y_train)
    y_predict = least_squares.predict(x_test)
    
    tol = len(y_test)
    score = 0
    for i in range(tol):
        if y_test[i] == y_predict[i]:
            score += 1

    
    print("least squares precision:{}%\n".format(float(score)/tol*100))
