# 机器学习算法：分类
*基于mnist数据集测试的线性分类算法的实现*
---

## 数据集简介
### 1. 来源
>  http://yann.lecun.com/exdb/mnist/

也可以直接通过sklearn.dataset.load_digits()导入。

### 2. 简介
*mnist数据集*
手写图片数据集。每张图片以28*28像素的矩阵表示。具体的说明参见上述网址

我们的模型目标即，根据这个28*28的矩阵，对图片进行分类


## 项目组成
### 0. file_io
*这里直接使用了1 knn的数据处理模块，不再赘述*
``` python
    decode_idx3(idx3_file_path) # 读取mnist数据集中idx3格式文件，转换成numpy.array的格式
    decode_idx1(idx1_file_path) # 读取mnist数据集中idx1格式文件，转换成numpy.array的格式

    load_trainset()             # 读取训练集
    load_testset()              # 读取测试集
                                # 两部分数据集无交叉部分

```
### 1. pre_work
*数据预处理模块，包含了预处理函数，混合使用*
```python

    nor()           # 正则化
    std()           # 标准化
    sig()           # sigmoid
    squ()           # 平方
    mx()            # 平移至以0为中心
    gs()            # 高斯

```

### 2. Least_squares
*最小二乘法分类算法主体*
```python

    __init__()                      # 类的构造函数
    fit(self, x, y)                 # 使用训练集训练模型
    predict(x)                      # 使用训练好的模型，将x分类

```


### 3. Fisher
*Fisher算法主体*
```python

    __init__(k)                        # 类的构造函数，目标是将至k维
    tranform(x,y)                      # 根据x与标签y，将数据降维

```


### 4. Perceptron
*感知器模型进行二分类的算法主体*
```python

    __init__()                           # 类的构造函数
    fit(self, x, y, lmd = 1, k = 200)    # 使用训练集训练模型，可自定义参数
    predict(x)                           # 使用训练好的模型，根据x预测y并返回
                                         # 注：该模型仅能用于二分类

```


## 测试结果

截止22/11/1，代码测试无误
