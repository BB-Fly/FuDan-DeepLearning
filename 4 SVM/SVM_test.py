from sklearn.svm import SVC,LinearSVC,NuSVC
from file_io import *

train_x,train_y,test_x,test_y = mnist()

train_x = normalization(train_x)
test_x = normalization(test_x)


svc = SVC(kernel='sigmoid')
lsvc = LinearSVC()
nsvc = NuSVC(kernel='sigmoid')

svc.fit(train_x,train_y)
lsvc.fit(train_x,train_y)
nsvc.fit(train_x,train_y)

y1 = svc.predict(test_x)
y2 = lsvc.predict(test_x)
y3 = nsvc.predict(test_x)

s1 = score(y1,test_y)
s2 = score(y2,test_y)
s3 = score(y3,test_y)

print(s1,s2,s3)