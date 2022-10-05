# 基于线性回归的机器学习算法
*基于波士顿房价数据集测试的简单线性回归算法的实现*
---

## 数据集简介
### 1. 来源
>  https://www.kaggle.com/vikrishnan/boston-house-prices

也可以直接通过sklearn.dataset.load_boston()导入。

### 2. 简介
*波士顿房价数据集*
数据集共包含506条数据，每条数据可以看做一个14×1的向量。前13个数值表示房子相关的13个属性，即x；最后一个为房价，即y。
x包含13个属性：CRIM、ZN、INDUS、CHAS、NOX、AGE、DIS、RAD、TAX、RTRATIO、B-1000、LSTAT、MEDV，含义可参照上述链接。

我们的模型目标即，根据这13个属性，预测y的值


## 项目组成
### 0. pre_vsl
*可视化数据集，直观展现x的每个属性和y的关系，本身并不参与模型的建立*

    pre_vsl()   # 读取boaston数据集，分别以可视化的方式展示x的13个属性和y的关系
                # 执行结果参考下图：

![Image](https://github.com/BB-Fly/FuDan-2022-DeepLearning/blob/main/hw2/img/pre_vsl.jpg)

### 1. pre_work
*数据集文件的读取与格式化模块*

    load_trainset() # 读取并截取数据集中的一部分作为训练集使用
    load_testset()  # 读取并截取数据集中的一部分作为测试集使用
                    # 两部分数据集无交叉部分

### 2. linear
*普通线性回归算法主体*

    __init__()                      # Linear类的构造函数
    fit(self, x, y)                 # 使用训练集训练模型
    predict(x)                      # 使用训练好的模型，根据x预测y并返回

对波士顿房价数据集的预测结果如图：
![Image](https://github.com/BB-Fly/FuDan-2022-DeepLearning/blob/main/hw2/img/linear.jpg)

### 3. ridge
*岭回归算法主体*

    __init__()                      # Ridge类的构造函数
    fit(self, x, y, lmd = 0.2)      # 使用训练集训练模型，可自定义lmd参数
    predict(x)                      # 使用训练好的模型，根据x预测y并返回

对波士顿房价数据集的预测结果如图：
![Image](https://github.com/BB-Fly/FuDan-2022-DeepLearning/blob/main/hw2/img/ridge.jpg)

### 4. lasso
*lasso回归算法主体*

    __init__()                                                              # Lasso类的构造函数
    fit(self, x, y, lmd = 0.2, learning_rate = 0.000005, epochs = 50000)    # 使用训练集训练模型，可自定义参数
    predict(x)                                                              # 使用训练好的模型，根据x预测y并返回

对波士顿房价数据集的预测结果如图：
![Image](https://github.com/BB-Fly/FuDan-2022-DeepLearning/blob/main/hw2/img/lasso.jpg)

### 5. test
分别使用sklearn内置的3种回归算法，以及自编的上述3种回归算法，在波士顿房价数据集上进行预测。
展示可视化的预测结果。
并使用sklearn的打分函数评价回归的结果

## 测试结果

截止22/10/6，代码测试无误

![Image](https://github.com/BB-Fly/FuDan-2022-DeepLearning/blob/main/hw2/img/test.jpg)

![Image](https://github.com/BB-Fly/FuDan-2022-DeepLearning/blob/main/hw2/img/score.jpg)

从折线图上来看，几种回归方法都有着不错的预测效果。
从打分结果来看，针对此数据集，自编的Ridge回归效果最好；除了自编的Lasso回归以外，其他回归的分数相近。

但自编的Lasso回归使用了梯度下降的方法，在参数的调整上还有很大的进步空间。