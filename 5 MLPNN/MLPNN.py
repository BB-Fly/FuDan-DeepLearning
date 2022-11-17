import torch
from torch import nn
import file_io


class _mlpnn(nn.Module):
    def __init__(self, x_size):
        super(_mlpnn,self).__init__()

        self.linear1 = nn.Sequential(
            nn.Linear(x_size,256),
            nn.ReLU(True)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(256,64),
            nn.ReLU(True)
        )
        self.linear3 = nn.Sequential(
            nn.Linear(64,10),
            #nn.Sigmoid()
        )
 
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

class MLPNN():
    def __init__(self,epochs = 100,loss_func = nn.CrossEntropyLoss()):
        self.loss_func = loss_func
        self.epochs = epochs


    def fit(self,X,Y):
        self.net = _mlpnn(X.shape[1])

        trainer = torch.optim.SGD(self.net.parameters(), lr=0.05)

        for i in range(self.epochs):
            outX = self.net(X)
            trainer.zero_grad()
            loss = self.loss_func(outX, Y)
            loss.backward()
            trainer.step()

    
    def predict(self,X):
        outX = self.net(X)
        return outX

            



if __name__ == '__main__':

    train_x, train_y, test_x, test_y = file_io.mnist_tensor()

    mlpnn = MLPNN(loss_func=nn.MultiLabelSoftMarginLoss())
    mlpnn.fit(train_x,train_y)

    predict_y = mlpnn.predict(test_x)

    print(file_io.score(file_io.from_label(predict_y),file_io.from_label(test_y)))