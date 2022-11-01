import numpy as np
import heapq
import copy
import matplotlib.pyplot as plt


# 暴力kNN算法
class kNN():
    def __init__(self, train_set, test_set):
        self.test_set = test_set
        self.train_set = train_set
        self.train_num=len(self.train_set[0])
        self.test_num=len(self.test_set[0])
        self.dists=[]


    # 为避免重复工作，进行预处理
    def pre_func(self, right):
        for i in range(0,self.test_num):
            dists_t=[]
            for j in range(0, self.train_num):
                dist = np.sum((self.train_set[0][j]-self.test_set[0][i])**2)
                # dist = np.sum(abs(self.train_set[0][j]-self.test_set[0][i]))
                heapq.heappush(dists_t,[-1*dist, int(self.train_set[1][j])])
                if(len(dists_t)>right):
                    heapq.heappop(dists_t)
            self.dists.append(dists_t)


    # 返回在给定的数据集、训练集下，取定k值时，准确率为多少
    def func(self, k):
        hit=0
        for i in range(0,self.test_num):
            dists=copy.copy(self.dists[i])
            score = np.zeros(10,int)
            while(len(dists)>k):
                heapq.heappop(dists)
            while(len(dists)>0):
                score[heapq.heappop(dists)[1]] += 1
            t_idx=0
            for j in range (0, 10):
                if score[t_idx] < score[j]:
                    t_idx=j
            if t_idx==self.test_set[1][i]:
                hit+=1
        precision = float(hit)/float(self.test_num)
        print("k=",k,", precision is ",precision,"\n")
        return precision


    # 寻找在left-right范围内最适合（准确率最高）的整数k
    def find_best_k(self, left, right, draw = False):
        score = np.zeros(right-left+1,dtype=float)
        for i in range(left,right+1):
            score[i-left] = self.func(i)
        t_idx=0
        for i in range(left, right+1):
            if score[t_idx]<score[i-left]:
                t_idx=i-left
        if draw:
            plt.scatter(range(left, right+1), score)
            plt.show()

        return t_idx+left, score[t_idx]