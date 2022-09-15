from turtle import left
from kNN import kNN
import os
# import torchvision.datasets.mnist as mnist
from file_io import *


# 数据集所在的根目录，需要根据设备更改
root = "C:\\Users\\86137\\Desktop\\study\\DeepLearning\\lab\\dataset\\mnist\\"

# train_set = (
#     mnist.read_image_file(os.path.join(root, 'train-images.idx3-ubyte')).reshape(60000,784).numpy()[0:60000:120],
#     mnist.read_label_file(os.path.join(root, 'train-labels.idx1-ubyte')).numpy()[0:60000:120]
# )
# test_set = (
#     mnist.read_image_file(os.path.join(root, 't10k-images.idx3-ubyte')).reshape(10000, 784).numpy()[0:10000:10],
#     mnist.read_label_file(os.path.join(root, 't10k-labels.idx1-ubyte')).numpy()[0:10000:10]
# )
train_set = (
    decode_idx3(root+'train-images.idx3-ubyte').reshape(60000,784)[0:60000:5],
    decode_idx1(root+'train-labels.idx1-ubyte')[0:60000:5]
)
test_set = (
    decode_idx3(root+'t10k-images.idx3-ubyte').reshape(10000, 784)[0:10000:40],
    decode_idx1(root+'t10k-labels.idx1-ubyte')[0:10000:40]
)

knnNet = kNN(train_set, test_set)

# for test
# score=knnNet.func(5)
l = 1
r = 20
knnNet.pre_func(r)

k, precision= knnNet.find_best_k(l,r, True)

print("the precision is ",100*precision,"% while k is ",k)

