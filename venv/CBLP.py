# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from copy import deepcopy

from arrayOperation import climbAlgorithm,array_change_percent,findAdaptThreshold,A2onehotA
from drawImage import drawNetwork,drawLabel

class CBLP():
    def __init__(self,model,percent=0.1,topN=0,):
        self.model=model
        self.percent=percent
        self.topN=topN

    def fit(self,propertysMatrix,):
        self.n = np.shape(propertysMatrix)[0]
        EDA = squareform(pdist(propertysMatrix, metric='seuclidean'))
        # ============计算标准欧式距离============================
        SIA = 1 / (EDA + 1)
        SIA=(SIA- np.min(SIA)) / (np.max(SIA) - np.min(SIA))
        for i in range(self.n):
            SIA[i][i] = 0
        # ===================计算相似性 1/(x+1),再标准化=========================
        if self.model=='even':
            if self.percent==0:
                return np.zeros(self.n).astype(np.int8)
            self.threshold=np.sort(np.reshape(SIA,(-1)))[int((1-self.percent)*self.n*self.n)]
            WA = (SIA > self.threshold) * SIA
            # ===================保留前percent的值=========================
        elif self.model=='uneven' and self.topN<>0:
            self.threshold=findAdaptThreshold(SIA)
            WA = (SIA > self.threshold) * SIA
            argindex = np.argsort(-SIA, axis=0)
            index1 = np.reshape(argindex[:self.topN], (1, -1))
            index2 = np.tile(np.arange(self.n), self.topN)
            index = index1, index2
            topNLINKA = np.zeros(np.shape(SIA))
            topNLINKA[index] = 1
            topNLINKA = topNLINKA + topNLINKA.T
            topNLINKA = (topNLINKA > 0) * 1
            WA[np.where(WA == 0)] = (SIA * topNLINKA)[np.where(WA == 0)]
            # ===================基于前percent的值，加上topN的关系=========================
        else:
            print('need true model parameter')
            return np.zeros(self.n).astype(np.int8)
        # ======================过滤相似性，得到权重矩阵==========================
        # drawNetwork(propertysMatrix, WA, title="network", withId=False)
        seedList = climbAlgorithm(WA, int(0.1 * self.n))

        LabelA = np.zeros((self.n, len(seedList)))
        for i in range(len(seedList)):
            LabelA[seedList[i]][i] = 1
        # drawLabel(propertysMatrix, LabelA=A2onehotA(LabelA), t='seed size:'+str(len(seedList)),)
        # ===================爬出种子集合=========================
        LabelA = np.zeros((self.n, len(seedList)))
        for i in range(len(seedList)):
            LabelA[seedList[i]][i] = 1
        # ==========================基于种子集合初始化标签向量矩阵================
        for t in range(50):
            LabelA, changed = self.updateLabelA(WA, LabelA)
            # =============更新标签矩阵=======================================
            # if t == 5:
            #     LabelA = self.cleanSmallLabel(LabelA, k=len(seedList))
            # ==========================大致收敛后，减去太小的社区=====================
            # drawLabel(propertysMatrix, LabelA=A2onehotA(LabelA), t='iteration time:' + str(t))

            if changed < 0.0001 and t > 5:
                # print('iteration stop in ', t)
                break

        # ============================标签传播至收敛===========================
        size = np.shape(LabelA)[0]
        maxIndex = np.argmax(LabelA, axis=1)
        zeroIndex = np.argwhere(np.sum(LabelA, axis=1) == 0)
        LabelA[:] = 0
        LabelA[np.arange(size), maxIndex] = 1
        LabelA[zeroIndex, :] = 0
        self.labels_ = np.zeros(self.n)
        index = np.where(LabelA == 1)
        self.labels_[index[0]] = index[1]
        return self

    def updateLabelA(self,WA, LabelA):
        LINKA = (WA > 0) * 1
        DEGA = np.sum(LINKA, axis=0)
        sortIdA = np.argsort(DEGA)
        LabelA1 = deepcopy(LabelA)
        for id in sortIdA:
            newLabelFloat = np.dot(WA[id], LabelA)
            if np.sum(newLabelFloat) == 0.0:
                continue
                # ==========邻接点都没有标签的情况，直接跳过=========
            # newLabelFloat=newLabelFloat
            newLabelFloat = newLabelFloat + 1-np.e **(-(np.dot(LINKA[id] * DEGA, LabelA)))

            # =====影响标签因素：标签权重和，标签所在邻接点的度的和=============
            maxIndex = np.argmax(newLabelFloat)
            LabelA1[id][:] = 0
            LabelA1[id][maxIndex] = 1
        #     ===============每行最大的那个值赋值为1，其他为0===========
        changed = array_change_percent(LabelA, LabelA1)
        return LabelA1, changed

    def cleanSmallLabel(self,LabelA, k):
        n = np.shape(LabelA)[0]
        labelNums = np.sum(LabelA, axis=0)
        min_num = n / k
        # min_num=20
        badCindex = np.where(labelNums < min_num)
        if np.sum(LabelA[:, badCindex]) > 0:
            LabelA[:, badCindex] = 0
        return LabelA

