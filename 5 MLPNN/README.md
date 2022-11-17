# MLPNN
*基于MNIST数据集测试的SVC分类*
---
## 数据集简介
### 1. 来源

>  http://yann.lecun.com/exdb/mnist/


### 2. 简介

*MNIST数据集*
手写图片数据集。每张图片以28*28像素的矩阵表示。具体的说明参见上述网址


## 项目组成
### 0. file_io
*数据的预处理模块*
``` python

    mnist_tensor()         # 读取mnist数据集，返回train_x，train_y，test_x，test_y

```

### 1. MLPNN
*MLPNN的pytorch实现*
``` python
    '''how to use class:MLPNN '''
    mlpnn = MLPNN(loss_func=nn.MultiLabelSoftMarginLoss())
    mlpnn.fit(train_x,train_y)

    predict_y = mlpnn.predict(test_x)

```




截止2022/11/17, 测试结果无误
