from random import randint
import numpy as np
import file_io
from sklearn.svm import SVC


class SVM:
    def __init__(self,C=20,e=0.01,kernal = 0,echo=1000) -> None:
        self.C = C
        self.e = e
        self.kernal = kernal
        self.echo = echo
        pass

    def fit(self,x,y):
        self.X = np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
        self.Y = np.array(y)

        self.alpha = np.random.randint(0,self.C,size=self.X.shape[0])
        self.alpha = np.array(self.alpha,dtype=float)
        self.bias = randint(-10,10)

        for i in range(self.echo):
            index1, index2 = self.outer_loop()
            if index1 == -1:
                break
            train_result = self.update(index1, index2)
            if train_result == True:
                break   

    def get_kernal(self,x1,x2,sigma = 0.5):
        if self.kernal==0:
            return np.exp(np.linalg.norm(x1-x2)/(-2*sigma*sigma))
        else:
            return abs(sum(x1*x2))**0.5

    def get_kernal_array(self, x):
        temp = np.zeros(self.X.shape[0])
        for idx in range(self.X.shape[0]):
            temp[idx] += self.get_kernal(x,self.X[idx])
        return temp

    def func(self, x):
        result = np.sum(self.get_kernal_array(x) * self.Y * self.alpha) + self.bias
        return result

    def inner_loop(self, index):
        result_index = 0
        temp_error = 0
        for i in range(self.alpha.shape[0]):
            diff_error = np.abs(self.func(self.X[index])-self.Y[index]-self.func(self.X[i])+self.Y[i])
            temp = self.get_kernal(self.X[index],self.X[index]) + self.get_kernal(self.X[i],self.X[i]) - 2 * self.get_kernal(self.X[index],self.X[i])
            if diff_error > temp_error and i != index and temp > 0:
                temp_error = diff_error
                result_index = i
        return result_index

    def outer_loop(self):
        for i in range(self.X.shape[0]):
            # 选择违反KKT原则最严重的alpha
            if 0 < self.alpha[i] < self.C:
                if self.Y[i] * self.func(self.X[i]) != 1:
                    index2 = self.inner_loop(i)
                    return i, index2
            elif self.alpha[i] == 0:
                if self.Y[i] * self.func(self.X[i]) < 1:
                    index2 = self.inner_loop(i)
                    return i, index2
            elif self.alpha[i] == self.C:
                if self.Y[i] * self.func(self.X[i]) > 1:
                    index2 = self.inner_loop(i)
                    return i, index2
        return -1, -1

    def update(self, index1, index2):
        # 计算运算量
        old_alpha = self.alpha.copy()
        x1 = self.X[index1]
        y1 = self.Y[index1]
        x2 = self.X[index2]
        y2 = self.Y[index2]
        K11 = self.get_kernal(x1,x1)
        K22 = self.get_kernal(x2,x2)
        K12 = self.get_kernal(x1,x2)
        e1 = self.func(self.X[index1]) - self.Y[index1]
        e2 = self.func(self.X[index2]) - self.Y[index2]
        # 确定alpha范围
        if y1 != y2:
            L = max(0, old_alpha[index2] - old_alpha[index1])
            H = min(self.C, self.C + old_alpha[index2] - old_alpha[index1])
        else:
            L = max(0, old_alpha[index1] + old_alpha[index2] - self.C)
            H = min(self.C, old_alpha[index1] + old_alpha[index2])


        # 运算alpha2
        alpha2 = old_alpha[index2] + y2 * (e1 - e2) / (K11 + K22 - 2 * K12)
        # 确定是否在范围内
        if alpha2 < L:
            alpha2 = L
        elif alpha2 > H:
            alpha2 = H
        # 计算alpha1
        alpha1 = old_alpha[index1] + y1 * y2 * (old_alpha[index2] - alpha2)

        # 更新alpha
        self.alpha[index1] = alpha1
        self.alpha[index2] = alpha2

        # 计算b
        b1 = -e1 - y1 * K11 * (alpha1 - self.alpha[index1]) - y2 * K12 * (alpha2 - self.alpha[index2]) + self.bias
        b2 = -e2 - y1 * K12 * (alpha1 - self.alpha[index1]) - y2 * K22 * (alpha2 - self.alpha[index2]) + self.bias

        # 更新b
        if 0 < alpha1 < self.C:
            self.bias = b1
        elif 0 < alpha2 < self.C:
            self.bias = b2
        else:
            self.bias = (b1 + b2) / 2

        if np.linalg.norm(old_alpha - self.alpha) < self.e:
            return True
        else:
            return False

    def predict(self,x):
        X = np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
        Y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            if self.func(X[i])>0:
                Y[i] = 1
            else:
                Y[i] = -1
        
        return Y





if __name__ =="__main__":
    x_train, y_train, x_test, y_test = file_io.mnist()
    x_train = np.array(x_train[::50])
    y_train = np.array(y_train[::50])
    x_test = np.array(x_test[::5])
    y_test = np.array(y_test[::5])

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

    #s = SVM(kernal=1)
    s = SVC(kernel="linear")
    s.fit(x_train, y)
    Y_predict = s.predict(x_test)

    cnt = 0
    for i in range(len(Y)):
        if int(Y[i])==int(Y_predict[i]):
            cnt += 1
    
    print("SVM precision:{}%\n".format(float(cnt)/len(Y)*100))

