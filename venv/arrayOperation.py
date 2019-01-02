# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
from time import clock
from copy import deepcopy
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from math import factorial

import drawImage
MAX_THRESHOLD=1
MAX_CLUSTER_NUM=30

NOISE_THRESHOLD=0.2
SIA_ADAPT_PARAMETER=5
THRESHOLD_ADAPT_P=0.05

def SIA2WA(SIA,WA2SIAway='norm',topN=0,percent=1,onlyTopN=False):
    n = np.shape(SIA)[0]
    if percent<1:
        if percent==0:
            return np.zeros(np.shape(SIA))
        th=np.reshape(SIA,(-1))
        th=np.sort(th)
        th=th[int((1-percent)*n*n)]
        WA=filter(SIA,th)
    else:
        if WA2SIAway=='norm':
            # 标准化
            m = np.mean(SIA)
            s = np.std(SIA)
            SIA=(SIA-m)/s
            # m = np.max(SIA, axis=0) - 0.25
            LINKA = (SIA > 0.5) * 1

            WA = LINKA * SIA
        elif WA2SIAway=='adapt':
            # 自适应阈值 特异性阈值
            adaptThreshold = findAdaptThreshold(SIA)
            # LINKA = adaptSIA(SIA, threshold=adaptThreshold)
            LINKA=(SIA>adaptThreshold)*1
            WA = LINKA * SIA
        if topN>0:
            topN = topN
            argindex = np.argsort(-SIA, axis=0)
            index1 = np.reshape(argindex[:topN], (1, -1))
            index2 = np.tile(np.arange(n), topN)
            index = index1, index2
            topNLINKA = np.zeros(np.shape(SIA))
            topNLINKA[index] = 1
            topNLINKA = topNLINKA + topNLINKA.T
            topNLINKA = (topNLINKA > 0) * 1
            WA[np.where(WA == 0)] = (SIA * topNLINKA)[np.where(WA == 0)]
            if onlyTopN:
                WA=topNLINKA*SIA
    return WA
def topNSIA(SIA,N):
    topN=N
    n= np.shape(SIA)[0]
    argindex = np.argsort(-SIA, axis=0)
    index1 = np.reshape(argindex[:topN], (1, -1))
    index2 = np.tile(np.arange(n), topN)
    index = index1, index2
    topNLINKA = np.zeros(np.shape(SIA))
    topNLINKA[index] = 1
    topNLINKA = topNLINKA + topNLINKA.T
    topNLINKA = (topNLINKA > 0) * 1
    return topNLINKA * SIA
# 爬山贪婪算法不断寻找边际影响范围最大的节点,返回 种子列表和大小
def climbAlgorithm(WA,k):

    LINKA = (WA > 0) * 1
    pLINKA = deepcopy(LINKA)
    pDEGA = np.sum(LINKA,axis=0)
    seedList = []
    for i in range(k):
        # 寻找度最大的节点ID，若有多个，随机选择一个
        maxDeg_id = findMaxDegSimilar_id(WA,pDEGA)
        if maxDeg_id == -1:
            break
        seedList.append(maxDeg_id)
        # 更新可传播度矩阵，包括直连的节点度减1，间接连接（间隔一节点的）节点的可传播度减少
        # print pDEGA
        pLINKA, pDEGA = updatepDEGA(pLINKA, maxDeg_id)
    # print('seed list size:' + str(len(seedList)))
    return seedList
# 寻找影响力最大的节点
def seedByA(A,k):
    n=np.shape(A)[0]
    size=n/10 if n/10>0 else n
    A=np.sort(-A,axis=0)[:size]
    ASum=np.sum(A,axis=0)
    seedList=np.argsort(ASum)[:k]
    seedList=seedList.tolist()
    return seedList
def updateLabelA(WA,LabelA):
    LINKA = (WA > 0) * 1
    DEGA = np.sum(LINKA, axis=0)
    sortIdA = np.argsort(DEGA)
    LabelA1 = deepcopy(LabelA)
    for id in sortIdA:
        newLabelFloat=np.dot(WA[id],LabelA)
        if np.sum(newLabelFloat)==0.0:
            continue
            # ==========邻接点都没有标签的情况，直接跳过=========
        # newLabelFloat=newLabelFloat
        newLabelFloat=newLabelFloat+f1(np.dot(LINKA[id]*DEGA,LabelA))
        # =====影响标签因素：标签权重和，标签所在邻接点的度的和=============
        LabelA1[id]=max2one(newLabelFloat)
    changed =array_change_percent(LabelA, LabelA1)
    return LabelA1,changed
def cleanSmallLabel(LabelA,k):
    n=np.shape(LabelA)[0]
    labelNums = np.sum(LabelA, axis=0)

    min_num=n/k
    # min_num=20
    badCindex = np.where(labelNums < min_num)
    if np.sum(LabelA[:, badCindex]) > 0:
        LabelA[:, badCindex] = 0
    return LabelA

# 矩阵阈值过滤
def filter(A,threshold=MAX_THRESHOLD):
    A=(A > threshold) * A
    return A
def findAdaptThreshold(SIA,threshold=MAX_THRESHOLD):
    size=np.shape(SIA)[0]
    t=threshold
    while(True):
        A = filter(SIA,t)
        LINKA=(A>0)*1
        DEGA=np.sum(LINKA,axis=0)
        percent=np.sum(DEGA==0)/float(size)
        # print percent-NOISE_THRESHOLD,t
        if percent<=NOISE_THRESHOLD:
            break
        else:
            t=t-t*THRESHOLD_ADAPT_P
    # print np.sum(DEGA == 0), size, percent, t
    print("the threshold is adapted to",t)
    return t
def adaptSIA(SIA,threshold=None):
    # degVar(SIA, threshold=threshold)
    if threshold ==None:
        threshold=findAdaptThreshold(SIA)
    size = np.shape(SIA)[0]
    WA = filter(SIA,threshold)
    LINKA=(WA>0)*1
    DEGA = np.sum(LINKA, axis=0)
    temp=-1/(1+SIA_ADAPT_PARAMETER*np.e**(DEGA))-0.01
    # temp = -1 / (5*DEGA+6)
    temp=np.tile(temp,size)
    temp=np.reshape(temp,(size,size))

    th=threshold*(temp+temp.T)+threshold
    LINKA=(SIA>th)*1
    return LINKA
def printDegVar(WA):
    LINKA=(WA>0)*1
    DEGA = np.sum(LINKA, axis=0)
    print('mean:',np.mean(DEGA),'cov:',np.cov(DEGA), 'var:',np.var(DEGA))
    return




def findMaxDeg_id(DEGA):
    degThreshold=1
    # degThreshold=np.log(np.shape(DEGA)[0])-2
    maxDeg = np.max(DEGA)

    if maxDeg<=degThreshold:
        # print 'no node degree greater than',degThreshold
        return -1

    maxDegIndex = np.where(DEGA == maxDeg)
    maxDeg_id=np.random.choice(maxDegIndex[0])
    # maxDeg_id=8
    # print (maxDeg_id,'be chosen to seed set,its propagation degree is ',maxDeg)
    return maxDeg_id
def findMaxDegSimilar_id(WA,DEGA):
    degThreshold=1
    # degThreshold=np.log(np.shape(DEGA)[0])-2
    maxDeg = np.max(DEGA)
    # 最近邻
    if maxDeg<=degThreshold:
        # print 'no node degree greater than',degThreshold
        return -1

    # 次近邻
    maxDegIndex = np.where(DEGA == maxDeg)

    SIA_Accumulation = np.sum(WA, 0)
    SIA_Accumulation=SIA_Accumulation*(DEGA==maxDeg)
    maxDeg_id=np.argmax(SIA_Accumulation)
    # maxDeg_id=8
    # print (maxDeg_id,'be chosen to seed set,its propagation degree is ',maxDeg,'SIA_Accumulation',np.max(SIA_Accumulation[maxDegIndex]))
    return maxDeg_id

def updatepDEGA(pLINKA,maxDeg_id):
    # 直连的节点可传播度减1
    maxDeg_id_nerbo = pLINKA[maxDeg_id]

    # 间接连接（间隔一节点的）节点的可传播度减少
    maxDeg_id_nerbo[maxDeg_id] = 1
    # print 'neibo\n',maxDeg_id_nerbo
    n_maxDeg_id_nerbo = np.tile(maxDeg_id_nerbo, (len(maxDeg_id_nerbo), 1))
    n_maxDeg_id_nerbo += n_maxDeg_id_nerbo.T
    n_maxDeg_id_nerbo = (n_maxDeg_id_nerbo > 0) * 1
    # drawImage.printDf(n_maxDeg_id_nerbo)
    decA = pLINKA * n_maxDeg_id_nerbo
    # print 'temp\n',decA
    pLINKA = pLINKA - decA
    # pLINKA[maxDeg_id]=[0]
    # pLINKA[:,maxDeg_id]=[0]
    # print pLINKA
    pDEGA=np.sum(pLINKA,axis=0)
    # print('Degree array after update:\n',pDEGA)
    return pLINKA,pDEGA

def ont_hot(A):
    onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
    integer_encoded = A.reshape(len(A), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return onehot_encoded
def array_change_percent(A,B):

    if np.shape(A)!=np.shape(B):
        print('shape of array error')
        return 0
    A_B = np.abs(A - B)
    temp=np.sum((A_B>0.005)*1)*1.0
    size=np.shape(A)[0]*np.shape(A)[1]
    return temp/size
def max2one(A):
    maxIndex = np.argmax(A)
    A[:] = 0
    A[maxIndex] = 1
    return A
def A2onehotA(A):
    size=np.shape(A)[0]
    maxIndex=np.argmax(A,axis=1)
    zeroIndex=np.argwhere(np.sum(A,axis=1)==0)
    A[:]=0
    A[np.arange(size),maxIndex]=1
    A[zeroIndex,:]=0
    return A
def f1(x):
    t=np.e**(-x)
    t=1-t
    return t
def LabelA2Cnum(LabelA):
    num=np.sum(LabelA,axis=0)
    index=np.where(num>0)
    return np.array(map(int,num[index]))
def LabelA2List(LabelA):
    num = np.sum(LabelA, axis=0)
    index = np.where(num > 0)
    Clist=[]
    for i in index[0]:
        l=np.argwhere(LabelA[:, i] > 0)
        l=np.reshape(l,(1,-1))
        Clist.append(l.tolist()[0])
    return Clist
def labelMatrix2List(labelMatrix):
    label=list(set(labelMatrix))
    Clist=[]
    for l in label:
        Clist.append(np.where(labelMatrix==l)[0].tolist())
    return Clist


def LabelA_onehot2num(LabelA):
    k = np.shape(LabelA)[1]
    temp = np.sum(LabelA * np.arange(k), axis=1)
    temp[np.where(np.sum(LabelA, axis=1) == 0)] = -1
    lList = list(map(int, temp))
    return lList
def getNeighbor2(LINKA,id):
    neighbor1=np.argwhere(LINKA[id]>0)
    neighbor2=np.sum(LINKA[neighbor1],axis=0)
    neighbor2=(neighbor2>0)*1
    return neighbor2
def PCAchange(A,n_components):
    pca = PCA(n_components=n_components)  # 这个参数可以指定你想要的维度, 也可以设置保留信息的数量 # 浮点数表示保留信息的比例 # 整数的话表示维度
    pca.fit(A)
    result = pca.transform(A)
    return result

def Normalize(A):
    return (A-np.min(A))/(np.max(A)-np.min(A))
# 距离矩阵转为相似度矩阵
def EDA2SIA(EDA):
    SIA = 1 / (EDA + 1)
    SIA=Normalize(SIA)
    for i in range(np.shape(EDA)[0]):
        SIA[i][i]=0
    return SIA

def computeNoise(WA):
    Sum=np.sum(WA,axis=0)
    num=np.shape(np.where(Sum==0.0))[1]
    n=np.shape(WA)[0]
    return num*1.0/n
def mylabelMatrix2normal(mylabelMatrix):
    Cnum=len(set(mylabelMatrix))
    list1=list(mylabelMatrix)
    c=[101,102,100]
    # [c.append(i) for i in list1 if not i in c]

    index=[]
    for i in c:
        l=np.where(mylabelMatrix==i)[0].tolist()

        index.append(l)
    for i in range(Cnum):
        mylabelMatrix[index[i]]=i+1
    return mylabelMatrix
# parameters' shape is n*1
def computeARI(labels_pred,labels_true,X=None):
    re=[]

    # label_encoder = LabelEncoder()
    # labels_pred = label_encoder.fit_transform(labels_pred)
    if np.shape(labels_pred)[0]== np.shape(labels_true)[0]:
        re.append(metrics.adjusted_rand_score(labels_true, labels_pred))
        re.append(metrics.adjusted_mutual_info_score(labels_true, labels_pred))
        re.append(metrics.v_measure_score(labels_true, labels_pred))
        re.append(metrics.fowlkes_mallows_score(labels_true, labels_pred))
        re.append(metrics.normalized_mutual_info_score(labels_true, labels_pred))
    # if X.all()!=None:
    #     re.append(metrics.calinski_harabaz_score(X, labels_pred))
    return re

def com(a,n):
    re=factorial(a)*factorial(n)
    re=re/factorial(n-a)
    return re

def mergeCluster(WA,mylabelMatrix):
    th=np.shape(WA)[0]/3
    myCnum = len(set(mylabelMatrix))
    myClist = labelMatrix2List(mylabelMatrix)

    CSIA = np.zeros((myCnum, myCnum))
    edgeNum = np.zeros((myCnum, myCnum))
    for i in range(myCnum):
        for j in range(myCnum):
            CSIA[i, j] = np.sum((WA[myClist[i]])[:, myClist[j]])
            LINKA = (WA > 0) * 1
            edgeNum[i, j] = np.sum((LINKA[myClist[i]])[:, myClist[j]])
            # if CSIA[i,j]!=0:
            #     CSIA[i,j]/=len(myClist[i])
            if j > i and CSIA[i, j] > 20:
                myClist[j].extend(myClist[i])
                myClist[i] = []
                break
    while [] in myClist:
        myClist.remove([])
    CSIA=np.around(CSIA,decimals=0)
    # print(CSIA)  # 输出最终结果
    return myClist2mylabelMatrix(myClist)

def myClist2mylabelMatrix(myClist):
    n=0
    while [] in myClist:
        myClist.remove([])
    for i in myClist:
        n+=len(i)
    mylabelMatrix=np.zeros(n)
    for i in range(len(myClist)):
        mylabelMatrix[myClist[i]]=i
    return mylabelMatrix.astype(np.int8)

def A2vectorA(A):
    A=mylabelMatrix2normal(A)
    Cnum=len(set(A))
    n=len(A)
    vectorA=np.zeros((n,Cnum))
    for i in range(n):
        vectorA[i,A[i]]=1
    return vectorA