# SVM & SVR
*基于波士顿房价数据集测试的SVR回归*
*基于MNIST数据集测试的SVC分类*
---

## 数据集简介
### 1. 来源
>  https://www.kaggle.com/vikrishnan/boston-house-prices

>  http://yann.lecun.com/exdb/mnist/


### 2. 简介
*波士顿房价数据集*
数据集共包含506条数据，每条数据可以看做一个14×1的向量。前13个数值表示房子相关的13个属性，即x；最后一个为房价，即y。
x包含13个属性：CRIM、ZN、INDUS、CHAS、NOX、AGE、DIS、RAD、TAX、RTRATIO、B-1000、LSTAT、MEDV，含义可参照上述链接。

我们的模型目标即，根据这13个属性，预测y的值

*MNIST数据集*
手写图片数据集。每张图片以28*28像素的矩阵表示。具体的说明参见上述网址


## 项目组成
### 0. file_io
*数据的预处理模块*

    mnist()         # 读取mnist数据集，返回train_x，train_y，test_x，test_y

    boston()        # 读取boaston数据集，返回train_x，train_y，test_x，test_y

    normalization() # 标准化数据


### 1. SVM
*SVM的手工实现*


### 2. SVM_test
*使用sklearn官方库的SVM分类mnist数据集*


### 3. SVR_test
*使用sklearn官方库的SVR预测Boston数据集*



截止2022/11/5, 测试结果无误