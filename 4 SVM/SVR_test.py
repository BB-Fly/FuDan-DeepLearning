from sklearn.svm import SVR, NuSVR, LinearSVR
from file_io import *
import matplotlib.pyplot as plt

train_x,train_y,test_x,test_y = boston()

train_x = normalization(train_x)
test_x = normalization(test_x)


svr = SVR(kernel='sigmoid')
lsvr = LinearSVR()
nsvr = NuSVR(kernel='sigmoid')

svr.fit(train_x,train_y)
lsvr.fit(train_x,train_y)
nsvr.fit(train_x,train_y)

y1 = svr.predict(test_x)
y2 = lsvr.predict(test_x)
y3 = nsvr.predict(test_x)

plt.figure()
plt.plot(test_y,label='True')
plt.plot(y1, label='svr')
plt.plot(y2, label='lsvr')
plt.plot(y3, label='nsvr')

plt.legend()
plt.show()