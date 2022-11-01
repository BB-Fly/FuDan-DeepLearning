import numpy as np
import file_io
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pre_work import *

class Fisher:
    def __init__(self,k=2):
        self.k = k
        self.eigVal = None
        self.eigVec = None
        self.Z = None

    def __covariance_matrix(self,X, Y=np.empty((0,0))):
        if not Y.any():
            Y = X
        n_samples = np.shape(X)[0]
        covariance_matrix = (1 / (n_samples-1)) * (X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0))

        return np.array(covariance_matrix, dtype=float)

    def __scatter_matrices(self,x,y):
        n_features = x.shape[1]
        labels = np.unique(y)

        SW = np.empty((n_features, n_features))
        SB = np.empty((n_features, n_features))

        x_mean = x.mean(axis = 0)

        for label in labels:
            Xi = x[y == label]
            SW = SW + (Xi.shape[0] - 1) * self.__covariance_matrix(Xi)

            Xi_mean = Xi.mean(axis=0)
            SB = SB + Xi.shape[0] * (Xi_mean - x_mean).dot((Xi_mean - x_mean).T)
        return SW, SB

    def transform(self, x, y):
        
        SW,SB = self.__scatter_matrices(x,y)

        U, S, V = np.linalg.svd(SW)
        S = np.diag(S)
        SW_inverse = V.dot(np.linalg.pinv(S)).dot(U.T)
        A = SW_inverse.dot(SB)

        self.eigVal, self.eigVec = np.linalg.eigh(A)

        idx = self.eigVal.argsort()[::-1]
        topk_eigen_values = self.eigVal[idx][:self.k]
        topk_eigen_vectors = self.eigVec[:, idx][:, :self.k]
        X_transformed = x.dot(topk_eigen_vectors)

        return X_transformed


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = file_io.mnist()

    colors = ['red','k','blue','m','y','w','g','c','brown','coral']

    
    #x = std(x_test)
    #x = nor(x_test)
    #x = sig(x_test)
    #x = x_test
    #x = squ(x_test)
    #x = mx(x_test)
    # x = nor(sig(x_test))
    x = gs(x_test)
    fisher = Fisher()
    y = fisher.transform(x,y_test)

    # fisher = LinearDiscriminantAnalysis(n_components=2)

    # y = fisher.fit_transform(normalize(x_test),y_test)

    plt.figure()
    for i in range(len(y)):
        plt.scatter(y[i][0],y[i][1],c=colors[int(y_test[i])])

    plt.show()
    pass