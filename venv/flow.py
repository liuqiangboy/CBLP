# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
from flow import *
from drawImage import *
from fileOperation import *
from arrayOperation import *

def runCBLP(fileName,iterationTime=30,WA2SIAway='norm',topN=0,percent=1,onlyTopN=False):
    propertysMatrix, labelMatrix = loadData(fileName)
    n = np.shape(propertysMatrix)[0]
    # ====================导入数据集=========================
    EDA = squareform(pdist(propertysMatrix, metric='seuclidean'))

    # ============计算标准欧式距离============================
    SIA = EDA2SIA(EDA)
    # printDegVar(SIA)

    # ===================计算相似性 1/(x+1),再标准化=========================
    WA = SIA2WA(SIA,WA2SIAway=WA2SIAway,topN=topN,percent=percent,onlyTopN=onlyTopN)
    # printDegVar(WA)
    # ======================过滤相似性，得到权重矩阵==========================
    start = clock()
    seedList = climbAlgorithm(WA, int(0.1 * n))
    # seedList=seedList+seedByA(SIA,int(0.1*n))
    # seedList=list(set(seedList))
    # ===========================得到种子集合===============================
    LabelA = np.zeros((n, len(seedList)))
    for i in range(len(seedList)):
        LabelA[seedList[i]][i] = 1
    # ==========================基于种子集合初始化标签向量矩阵================
    # drawLabel(propertysMatrix, LabelA=A2onehotA(LabelA), t=fileName.split('/')[-1].split('.')[0]+' seed size:'+str(len(seedList)),)
    # 画种子节点
    # drawNetwork(propertysMatrix, WA, title=fileName.split('/')[-1].split('.')[0], withId=False)
    # ===============画图，标出种子集合的点=============================

    for t in range(iterationTime):

        LabelA, changed=updateLabelA(WA,LabelA)
        # =============更新标签矩阵=======================================
        # print('iteration time :',t,"changed:",changed)

        # =============输出结果，画图======================================
        # if t==5:
        #     LabelA=cleanSmallLabel(LabelA,k=len(seedList))
        # ==========================大致收敛后，减去太小的社区=====================
        if changed<0.0001 and t>5:
            print('iteration stop in ',t)
            # drawLabel(propertysMatrix, LabelA=A2onehotA(LabelA), t=fileName.split('/')[-1] + ',time:' + str(t))
            break
        # drawLabel(propertysMatrix, LabelA=A2onehotA(LabelA), t=fileName.split('/')[-1] + ',time:' + str(t))
    # drawLabel(propertysMatrix, LabelA=A2onehotA(LabelA), t=fileName.split('/')[-1])
    # ============================标签传播至收敛===========================
    end = clock()
    print('time cost:', end - start)
    LabelA = A2onehotA(LabelA)
    mylabelMatrix=np.zeros(n)-1
    index=np.where(LabelA == 1)
    mylabelMatrix[index[0]] = index[1]
    # mylabelMatrix = mergeCluster(WA, mylabelMatrix)
    # drawLabel(propertysMatrix, LabelA=mylabelMatrix, t=fileName.split('/')[-1])
    myCnum = len(set(mylabelMatrix))

    # mylabelMatrix=mylabelMatrix2normal(mylabelMatrix)
    myClist=labelMatrix2List(mylabelMatrix)
    Cnum = len(set(labelMatrix))
    Clist = labelMatrix2List(labelMatrix)

    re = np.zeros((myCnum, Cnum))
    for i, l1 in enumerate(myClist):
        for j, l2 in enumerate(Clist):
            ret_list = list(set(l1) & set(l2))
            re[i][j] = len(ret_list)
    print('my   result', [len(myClist[i]) for i in range(myCnum)])
    print('true result', [len(Clist[i]) for i in range(Cnum)])
    print('ARI AMI V-measure FMI NMI:')
    print( computeARI(mylabelMatrix, labelMatrix,X=propertysMatrix))
    # print(re)
    # =================计算ARI值,输出结果============================


    print('================================='
          '======' + fileName.split('/')[-1] + '===============================')


def temp():
    drawNxNodes(propertysMatrix, withId=False)
    # LINKA=(WA>0)*1
    # DEGA=np.sum(LINKA,axis=0)
    # DEGA2=DEGA*DEGA
    # k2=np.mean(DEGA2)
    # k=np.mean(DEGA)
    # yizhi=k2/(k**2)

    # ==============输出每个类簇的个数=========================


    #
    #
    # re=np.zeros((len(myClist),Cnum))
    # for i,l1 in enumerate(myClist):
    #     for j,l2 in enumerate(Clist):
    #         ret_list = list(set(l1) & set(l2))
    #         re[i][j]=len(ret_list)
    # print(re)

