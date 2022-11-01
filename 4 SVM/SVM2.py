from colorsys import yiq_to_rgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import file_io


class SVM:
    def __init__(self, xSet, yArray, C=5, floatingPointError=0.0001):
        self.xMat = np.mat(xSet)  # (48,2)
        self.yMat = np.mat(yArray).T  # (48,1)
        self.rows = self.xMat.shape[0]
        self.cols = self.xMat.shape[1]
        self.alpha = np.mat(np.zeros(self.rows)).T  # (48,1)
        self.w = None  # 最后返回,计算过程不需要
        self.b = 0
        self.C = C  # C=None时表示hard margin
        self.fpe = floatingPointError

        self.trainCount = 0  # 记录训练次数
        self.K = np.matmul(self.xMat, self.xMat.transpose())
        # Ei 缓存
        self.EiCatch = np.zeros(self.rows)
        self.updateEi_catch()

    def predict(self, xArray):
        resultList = []
        for i in range(len(xArray)):
            v = np.sum(np.multiply(xArray[i], self.w)) + self.b
            if v > 0:
                resultList.append(1)
            else:
                resultList.append(-1)
        return resultList

    def score(self, xArray, yArray):
        resultList = self.predict(xArray)
        count = 0
        for i in range(len(yArray)):
            if int(resultList[i]) == int(yArray[i]):
                count += 1
        return round(count / len(yArray) * 100, 2)

    def train(self, maxCount, debug=False):
        self.trainCount = 0
        while self.trainCount < maxCount:
            self.update_allPoints(debug)
            self.trainCount += 1
        # 打印alpha信息
        if debug==True:
            print(self.alpha)

        return self.w, self.b

    def update_allPoints(self, debug=None):
        count = 0
        for alpha2_index in range(self.rows):
            if self.check_alpha2_needUpdate(alpha2_index):
                alpha1_index = self.selectAlpha1_index(alpha2_index)
                self.update_alpha_and_b(alpha1_index, alpha2_index)

                # 计算w
                self.w = np.matmul(np.multiply(self.yMat, self.alpha).T, self.xMat)
                if debug:
                    # 打印alpha信息
                    print(self.alpha)
                    # 画图
                    self.classifyDataAndPlot()
                    print("调整次数:{}".format(count + 1))
                    count += 1
                    # 打印ei信息
                    print(self.EiCatch)

    def check_alpha2_needUpdate(self, alpha2_index):
        Ei = self.EiCatch[alpha2_index]
        yi = self.yMat[alpha2_index, 0]
        alpha2 = self.alpha[alpha2_index, 0]
        fx = self.cal_Fx(alpha2_index)

        if alpha2 < 0 or alpha2 > self.C:
            return True

        if yi == 1 and fx >= 1:
            return False
        elif yi == -1 and fx <= -1:
            return False

        # 再来看看是否有足够的空间调整
        # Ei不为零的,alpha应该是0如果不是就要调整,alpha2调整量就是 -yi*Ei,如果是正的, alpha增加,但如果已经是C的话就不用处理了

        alpha2_change_direction = -yi * Ei
        if alpha2_change_direction > self.fpe and alpha2 < self.C:
            return True
        elif alpha2_change_direction < -self.fpe and alpha2 > 0:
            return True
        else:
            return False

    def update_alpha_and_b(self, alpha1_index, alpha2_index):
        alpha1_old = self.alpha[alpha1_index, 0]
        alpha2_old = self.alpha[alpha2_index, 0]
        y1 = self.yMat[alpha1_index, 0]
        y2 = self.yMat[alpha2_index, 0]

        alpha2_new_chiped = self.get_alpha2_new_chiped(alpha1_index, alpha2_index)
        alpha1_new = alpha1_old + y1 * y2 * (alpha2_old - alpha2_new_chiped)
        b_new = self.get_b_new(alpha1_index, alpha2_index, alpha1_new, alpha2_new_chiped)
        # 最后更新数据
        alpha2_new_chiped = round(alpha2_new_chiped, 5)
        alpha1_new = round(alpha1_new, 5)
        b_new = round(b_new, 5)

        self.alpha[alpha1_index, 0], self.alpha[alpha2_index, 0] = alpha1_new, alpha2_new_chiped
        self.b = b_new
        # 更新EiCatch
        self.updateEi_catch()
        return True

    def get_b_new(self, alpha1_index, alpha2_index, alpha1_new, alpha2_new_chiped):
        alpha1_old = self.alpha[alpha1_index, 0]
        alpha2_old = self.alpha[alpha2_index, 0]
        y1 = self.yMat[alpha1_index, 0]
        y2 = self.yMat[alpha2_index, 0]
        K11 = self.K[alpha1_index, alpha1_index]
        K12 = self.K[alpha1_index, alpha2_index]
        K22 = self.K[alpha2_index, alpha2_index]
        E1 = self.EiCatch[alpha1_index]
        E2 = self.EiCatch[alpha2_index]
        b1New = self.b - E1 + y1 * K11 * (alpha1_old - alpha1_new) + y2 * K12 * (alpha2_old - alpha2_new_chiped)
        b2New = self.b - E2 + y1 * K12 * (alpha1_old - alpha1_new) + y2 * K22 * (alpha2_old - alpha2_new_chiped)
        # 只有符合的alpha_new用来调整b
        if self.C is None:
            alpha1_valid = True if 0 < alpha1_new < self.fpe else False
            alpha2_valid = True if 0 < alpha2_new_chiped else False
        else:
            alpha1_valid = True if 0 < alpha1_new < self.C else False
            alpha2_valid = True if 0 < alpha2_new_chiped < self.C else False
        if alpha1_valid:
            b = b1New
        elif alpha2_valid:
            b = b2New
        else:
            b = (b1New + b2New) / 2
        return b

    def check_kkt_status(self):
        # yi和alpha的乘积和为0
        if not (-self.fpe < np.sum(np.multiply(self.yMat, self.alpha)) < self.fpe):
            return False
        # 然后检查每个alpha
        for i in range(len(self.alpha)):
            if self.check_satisfiy_kkt_onePoint(i) == False:
                return False
        return True

    def cal_Ei(self, index):
        v = self.cal_Fx(index) - self.yMat[index, 0]
        return round(v, 5)

    def cal_Fx(self, index):
        # (1,48) * (48,1)=1
        v = float(np.multiply(self.alpha, self.yMat).T * self.K[:, index] + self.b)
        return round(v, 5)

    def updateEi_catch(self):
        # alpha变动的时候更新
        for i in range(self.rows):
            v = self.cal_Ei(i)
            self.EiCatch[i] = v
        return True

    def check_alpha2_vaild(self, alpha1_index, alpha2_index, Ei_list):
        # 计算更新量是否足够
        if alpha1_index == alpha2_index:
            return False
        alpha2_new_chiped = self.get_alpha2_new_chiped(alpha1_index, alpha2_index, Ei_list)
        alpha2_old = self.alpha[alpha2_index, 0]
        if None == alpha2_new_chiped:
            return False
        else:
            if abs(alpha2_new_chiped - alpha2_old) > self.fpe:
                return True
            else:
                return False

    def get_alpha2_new_chiped(self, alpha1_index, alpha2_index):
        alpha2_old = self.alpha[alpha2_index, 0]
        y2 = self.yMat[alpha2_index, 0]
        E1 = self.EiCatch[alpha1_index]
        E2 = self.EiCatch[alpha2_index]
        eta = self.K[alpha1_index, alpha1_index] + self.K[alpha2_index, alpha2_index] - 2.0 * self.K[
            alpha1_index, alpha2_index]
        if eta == 0:
            return None
        try:
            alpha2_new_unc = alpha2_old + (y2 * (E1 - E2) / eta)
            alpha2_new_chiped = self.get_alpha2_chiped(alpha2_new_unc, alpha1_index, alpha2_index)
        except:
            print()

        return alpha2_new_chiped

    def get_alpha2_chiped(self, alpha2_new_unc, alpha1_index, alpha2_index):
        y1 = self.yMat[alpha1_index, 0]
        y2 = self.yMat[alpha2_index, 0]
        alpha1 = self.alpha[alpha1_index, 0]
        alpha2 = self.alpha[alpha2_index, 0]

        if self.C is None:
            # hard margin
            if y1 == y2:
                H = alpha1 + alpha2
                L = 0
            else:
                H = None
                L = max(0, alpha2 - alpha1)
        else:
            # soft margin
            if y1 == y2:
                H = min(self.C, alpha1 + alpha2)
                L = max(0, alpha1 + alpha2 - self.C)
            else:
                H = min(self.C, self.C - alpha1 + alpha2)
                L = max(0, alpha2 - alpha1)

        alpha2_new_chiped = None
        if alpha2_new_unc < L:
            alpha2_new_chiped = L
        else:
            if H is None:
                alpha2_new_chiped = alpha2_new_unc
            else:
                if alpha2_new_unc > H:
                    alpha2_new_chiped = H
                else:
                    alpha2_new_chiped = alpha2_new_unc
        return alpha2_new_chiped

    def selectJrand(self, i):
        j = i
        while i == j:
            j = int(np.random.uniform(0, self.rows))
        return j

    def selectAlpha1_index(self, alpha2_index):
        # 非零alpha的是sv的几率大
        E2 = self.EiCatch[alpha2_index]
        nonZeroList = []
        for i in range(self.rows):
            alpha = self.alpha[i, 0]
            if 0 < alpha < self.C:
                nonZeroList.append(i)

        if len(nonZeroList) == 0:
            return self.selectJrand(alpha2_index)
        else:
            maxDiff = 0
            j = -1
            for i in range(len(nonZeroList)):
                row = nonZeroList[i]
                if row == alpha2_index:
                    continue
                else:
                    E1 = self.EiCatch[row]
                    if abs(E1 - E2) > maxDiff:
                        maxDiff = abs(E1 - E2)
                        j = row
            if j == -1:
                return self.selectJrand(alpha2_index)
            else:
                return j

def runMySvm(xSet,ySet):
    classifier = SVM(xSet, ySet, C=2)

    # debug模式每次迭代更新一次图,可以看动画的效果
    w, b = classifier.train(100, debug=False)
    score = classifier.score(xSet, ySet)
    print("正确率:", score)
    #classifier.classifyDataAndPlot()

if __name__ =="__main__":
    x_train, y_train, x_test, y_test = file_io.mnist()
    x_train = np.array(x_train[::50])
    y_train = np.array(y_train[::50])
    x_test = np.array(x_test[::5])
    y_test = np.array(y_test[::5])

    y = np.array(y_train)
    Y = np.array(y_test)
    for i in range(len(y)):
        if y[i]%2==0:
            y[i] = 1
        else:
            y[i] = -1

    for i in range(len(Y)):
        if Y[i]%2==0:
            Y[i] = 1
        else:
            Y[i] = -1

    runMySvm(x_train,y)

    # s = SVM()
    # #s = SVC(kernel="linear")
    # s.fit(x_train, y)
    # Y_predict = s.predict(x_test)

    # cnt = 0
    # for i in range(len(Y)):
    #     if int(Y[i])==int(Y_predict[i]):
    #         cnt += 1
    
    # print("SVM precision:{}%\n".format(float(cnt)/len(Y)*100))