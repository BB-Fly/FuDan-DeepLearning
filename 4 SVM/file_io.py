import numpy as np
import struct
from sklearn import datasets


def decode_idx3(idx3_file):
    bin_data = open(idx3_file, 'rb').read()
    offset = 0
    fmt_header = '>iiii'
    _, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)

    image_size = num_rows*num_cols

    offset += struct.calcsize(fmt_header)
    fmt_image = '>'+str(image_size)+'B'
    images = np.empty((num_images, num_rows,num_cols))

    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image,bin_data, offset)).reshape(num_rows,num_cols)
        offset += struct.calcsize(fmt_image)

    return images

def decode_idx1(idx1_file):
    bin_data = open(idx1_file, 'rb').read()
    offset = 0
    fmt_header = '>ii'
    _, num_images = struct.unpack_from(fmt_header, bin_data, offset)

    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    images = np.empty(num_images)

    for i in range(num_images):
        images[i] = struct.unpack_from(fmt_image,bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)

    return images

def get_array(X,Y):
    X = np.array(X)
    Y = np.transpose(np.array(Y))
    return X,Y

def load_boston_dataset():
    # 获取数据集
    # x表示房屋属性，共13项，y表示房价
    train_x, train_y = datasets.load_boston(return_X_y = True)
    return get_array(train_x,train_y)

def mnist(root = "C:\\Users\\86137\\Desktop\\study\\DeepLearning\\FuDan-2022-DeepLearning\\0 dataset\\mnist\\"):
    # 需要根据文件在设备的路径获取数据
    train_x = decode_idx3(root+'train-images.idx3-ubyte').reshape(60000,784)[0:60000:12]
    train_y = decode_idx1(root+'train-labels.idx1-ubyte')[0:60000:12]

    test_x = decode_idx3(root+'t10k-images.idx3-ubyte').reshape(10000, 784)[0:10000:40]
    test_y = decode_idx1(root+'t10k-labels.idx1-ubyte')[0:10000:40]

    return train_x, train_y, test_x, test_y


def boston():
    x, y = load_boston_dataset()
    return x[0:450],y[0:450],x[451:506],y[451:506]


def normalization(X):
    l = X[0][0]
    r = X[0][0]
    for i in range(X.shape[0]):
        l = min(min(X[i]),l)
        r = max(max(X[i]),r)
    
    for i in range(X.shape[0]):
        X[i] = (r-X[i])/(r-l)

    return X

def score(Y_test,Y):
    a = len(Y)
    s = 0
    for i in range(a):
        if(Y_test[i]==Y[i]):
            s += 1

    return s/a
