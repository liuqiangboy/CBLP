# -*- coding: utf-8 -*-
import numpy as np
from fileOperation import loadData,fileList
from arrayOperation import *

def runCBLP(fileName,iterationTime=30,WA2SIAway='norm',topN=0,percent=1,onlyTopN=False):
    propertysMatrix, labelMatrix = loadData(fileName)
    n = np.shape(propertysMatrix)[0]
    # ====================导入数据集=========================
    EDA = squareform(pdist(propertysMatrix, metric='seuclidean'))
    # ============计算标准欧式距离============================
    SIA = EDA2SIA(EDA)

    # ===================计算相似性 1/(x+1),再标准化=========================
    WA = SIA2WA(SIA,WA2SIAway=WA2SIAway,topN=topN,percent=percent,onlyTopN=onlyTopN)
    # printDegVar(WA)
    # ======================过滤相似性，得到权重矩阵==========================
    seedList = climbAlgorithm(WA, int(0.1 * n))
    # ===========================得到种子集合===============================
    LabelA = np.zeros((n, len(seedList)))
    for i in range(len(seedList)):
        LabelA[seedList[i]][i] = 1
    # ==========================基于种子集合初始化标签向量矩阵================
    start = clock()
    for t in range(iterationTime):

        LabelA, changed=updateLabelA(WA,LabelA)
        # =============更新标签矩阵=======================================

        # =============输出结果，画图======================================
        if t==5:
            LabelA=cleanSmallLabel(LabelA,k=len(seedList))
        # ==========================大致收敛后，减去太小的社区=====================
        if changed<0.0001 and t>5:
            # print('iteration stop in ',t)
            break

    # ============================标签传播至收敛===========================
    end = clock()
    print('time cost:', end - start)
    LabelA = A2onehotA(LabelA)
    mylabelMatrix=np.zeros(n)-1
    index=np.where(LabelA == 1)
    mylabelMatrix[index[0]] = index[1]
    # =================计算ARI值,输出结果============================

