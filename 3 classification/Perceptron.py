import numpy as np
import file_io

class Perceptron:
    def __init__(self):
        pass

    def fit(self,x,y,lmd = 1, k = 200):
        m = x.shape[0]
        X = np.concatenate((np.ones((m,1)),x),axis=1)
        Y = y

        self.w = np.random.randint(-10,10,size=X.shape[1])
        self.w = np.array(self.w, dtype=float)

        cnt = 0
        while(cnt<k):
            tmp = X.dot(self.w)
            for i in range(m):
                if Y[i]*tmp[i]<0:
                    self.w += lmd*Y[i]*X[i]
            cnt += 1

    def predict(self, x):

        m = x.shape[0]
        X=np.concatenate((np.ones((m,1)),x),axis=1)

        Y = X.dot(self.w)
        for i in range(len(Y)):
            if Y[i]>=0:
                Y[i] = 1
            else:
                Y[i] =- 1
        return Y

if __name__ =="__main__":
    x_train, y_train, x_test, y_test = file_io.mnist()
    y = np.array(y_train)
    Y = np.array(y_test)
    for i in range(len(y)):
        if y[i]%2==0:
            y[i] = 1
        else:
            y[i] = -1

    for i in range(len(Y)):
        if Y[i]%2==0:
            Y[i] = 1
        else:
            Y[i] = -1

    perceptron = Perceptron()

    perceptron.fit(x_train, y)
    Y_predict = perceptron.predict(x_test)

    cnt = 0
    for i in range(len(Y)):
        if int(Y[i])==int(Y_predict[i]):
            cnt += 1
    
    print("Perceptron precision:{}%\n".format(float(cnt)/len(Y)*100))
    


