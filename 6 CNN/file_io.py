import numpy as np
import struct
import torch


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


def mnist(root = "C:\\Users\\86137\\Desktop\\study\\DeepLearning\\FuDan-2022-DeepLearning\\0 dataset\\mnist\\"):
    # 需要根据文件在设备的路径获取数据
    train_x = decode_idx3(root+'train-images.idx3-ubyte')[0:60000:10]
    train_y = decode_idx1(root+'train-labels.idx1-ubyte')[0:60000:10]

    test_x = decode_idx3(root+'t10k-images.idx3-ubyte')[0:10000:12]
    test_y = decode_idx1(root+'t10k-labels.idx1-ubyte')[0:10000:12]

    return train_x, train_y, test_x, test_y


def mnist_tensor(root="C:\\Users\\86137\\Desktop\\study\\DeepLearning\\FuDan-2022-DeepLearning\\0 dataset\\mnist\\"):
    train_x, train_y, test_x, test_y = mnist(root)
    train_y = to_label(train_y)
    test_y = to_label(test_y)
    return torch.from_numpy(np.array(train_x)).float(),torch.from_numpy(np.array(train_y)).float(),\
        torch.from_numpy(np.array(test_x)).float(),torch.from_numpy(np.array(test_y)).float()
    
def to_label(Y,l = 10):
    res = np.zeros(shape=(len(Y),l))
    for i in range(len(Y)):
        res[i][int(Y[i])] = 1.0
    return res


def from_label(Y):
    res = np.zeros(shape=len(Y))
    for i in range(Y.shape[0]):
        idx = 0
        for j in range(Y.shape[1]):
            if Y[i][idx]<Y[i][j]:
                idx = j
        res[i] = idx
    return res


def score(Y_test,Y):
    a = len(Y)
    s = 0
    for i in range(a):
        if(Y_test[i]==Y[i]):
            s += 1

    return s/a


if __name__ =='__main__':
    x1,y1,x2,y2 = mnist_tensor()
    pass