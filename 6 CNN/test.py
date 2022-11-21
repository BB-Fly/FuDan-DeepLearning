import torch
from torch import nn
import file_io
import numpy as np
import matplotlib.pyplot as plt


class _cnn(nn.Module):
    def __init__(self):
        super(_cnn,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1,10,5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10,20,5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.linear = nn.Sequential(
            nn.Linear(320,10),
            nn.LogSigmoid()
        )
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(1,-1)
        x = self.linear(x)
        return x

class CNN():
    def __init__(self,epochs = 10,loss_func = nn.CrossEntropyLoss()):
        self.loss_func = loss_func
        self.epochs = epochs


    def fit(self,X,Y,draw = False):
        self.net = _cnn()

        trainer = torch.optim.SGD(self.net.parameters(), lr=0.01,momentum=0.5)

        if draw:
            plt.figure()
            plt_x=[]
            plt_y=[]

        for i in range(self.epochs):
            for idx in range(X.shape[0]):
                outX = self.net(X[idx].reshape((1,28,28)))
                trainer.zero_grad()
                loss = self.loss_func(outX, Y[idx])
                loss.backward()
                trainer.step()
            if draw:
                plt_x.append(i)
                plt_y.append(loss.item())

        if draw:
            plt.plot(plt_x,plt_y)
            plt.show()

    
    def predict(self,X):
        outX = X.reshape((X.shape[0],1,X.shape[1],X.shape[2]))
        outX = self.net(outX)
        return outX

            



if __name__ == '__main__':

    train_x, train_y, test_x, test_y = file_io.mnist_tensor()

    #cnn = CNN(loss_func=nn.MultiLabelSoftMarginLoss())
    cnn = CNN(epochs=10)
    cnn.fit(train_x,train_y,False)

    predict_y = cnn.predict(test_x)

    print(file_io.score(file_io.from_label(predict_y),file_io.from_label(test_y)))