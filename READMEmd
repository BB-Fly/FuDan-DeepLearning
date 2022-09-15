# kNN算法
*基于mnist数据集测试的简单kNN算法实现*
---

## 数据集简介
### 1. 来源
> http://yann.lecun.com/exdb/mnist/

### 2. 简介
手写图片数据集。每张图片以28*28像素的矩阵表示。具体的说明参见上述网址


## 项目组成
### 1. file_io
*数据集文件的读取与格式化模块*

    decode_idx3(idx3_file_path) # 读取mnist数据集中idx3格式文件，转换成numpy.array的格式
    decode_idx1(idx1_file_path) # 读取mnist数据集中idx1格式文件，转换成numpy.array的格式

### 2. img_load
*将mnist数据集转换为图片与标签文档（不直接用于main程序）*

    img_main()  # 读取mnist数据集，分别将测试集与训练集图片提取到目标文件夹下
                # 并将标签写入目标文本文件下

### 3. kNN
*kNN算法主体*

    __init__(train_set, test_set)   # 类的构造函数，使用训练集与测试集初始化kNN网络
    pre_func(right)                 # right为k可能的最大值
                                    # 预处理后，网络将保存所有测试集向量最近的right个向量用于后续计算
    func(k)                         # 返回特定k值的情况下模型的准确率
    find_best_k(l,r)                # 在[l,r]范围内，寻找预测准确率最高的k值，返回它和对应的准确率

### 4. Main
*kNN算法在mnist数据集上预测实例*

## 测试结果

截止22/9/15，代码测试无误


当k取值11时，取得了最高为96.4%的预测准确率

