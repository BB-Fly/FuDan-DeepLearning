from turtle import left
from kNN import kNN
from file_io import *


# 数据集所在的根目录，需要根据设备更改
root = "C:\\Users\\86137\\Desktop\\study\\DeepLearning\\lab\\dataset\\mnist\\"

train_set = (
    decode_idx3(root+'train-images.idx3-ubyte').reshape(60000,784)[0:60000:5],
    decode_idx1(root+'train-labels.idx1-ubyte')[0:60000:5]
)
test_set = (
    decode_idx3(root+'t10k-images.idx3-ubyte').reshape(10000, 784)[0:10000:40],
    decode_idx1(root+'t10k-labels.idx1-ubyte')[0:10000:40]
)

knnNet = kNN(train_set, test_set)

l = 1
r = 20
knnNet.pre_func(r)

k, precision= knnNet.find_best_k(l,r, True)

print("the precision is ",100*precision,"% while k is ",k)

