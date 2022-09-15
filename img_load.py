import os
from skimage import io
import torchvision.datasets.mnist as mnist

def convert_to_img(train=True):
    root = "C:\\Users\\86137\\Desktop\\study\\DeepLearning\\lab\\dataset\\mnist"
    train_set = (
        mnist.read_image_file(os.path.join(root, 'train-images.idx3-ubyte')),
        mnist.read_label_file(os.path.join(root, 'train-labels.idx1-ubyte'))
    )
    test_set = (
        mnist.read_image_file(os.path.join(root, 't10k-images.idx3-ubyte')),
        mnist.read_label_file(os.path.join(root, 't10k-labels.idx1-ubyte'))
    )
    if train:
        f=open(root+'\\tarin.txt','w')
        data_path = root+'\\train\\'
        if(not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img, label) in enumerate(zip(train_set[0],train_set[1])):
            img_path=data_path+str(i)+'.jpg'
            io.imsave(img_path, img.numpy())
            f.write(img_path+' '+str(label)+'\n')
        f.close()
    else:
        f=open(root+'\\test.txt','w')
        data_path = root+'\\test\\'
        if(not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img, label) in enumerate(zip(test_set[0],test_set[1])):
            img_path=data_path+str(i)+'.jpg'
            io.imsave(img_path, img.numpy())
            f.write(img_path+' '+str(label)+'\n')
        f.close()

def img_main():
    convert_to_img(True)
    convert_to_img(False)

# img_main()