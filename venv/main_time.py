# -*- coding: utf-8 -*-
from CBLP import CBLP
from fileOperation import loadData
from arrayOperation import computeARI
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from time import clock
import numpy as np
import os
if __name__ =='__main__':
    evenPath = '/Users/liuqiang/Documents/标签传播过程论文/聚类数据集/even/'
    evenFileList=[]
    for i in os.listdir(evenPath):
        if i.split('.')[-1] == 'txt':
            evenFileList.append(evenPath + i)

    unevenPath = '/Users/liuqiang/Documents/标签传播过程论文/聚类数据集/uneven/'
    unevenFileList = []
    for i in os.listdir(unevenPath):
        if i.split('.')[-1] == 'txt':
            unevenFileList.append(unevenPath + i)
    unevenFileList = ['/Users/liuqiang/Documents/标签传播过程论文/聚类数据集/uneven/cancer.txt',]
    for filename in unevenFileList:
        X, labels_true = loadData(filename)
        k = len(set(labels_true))
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        hierarchical = AgglomerativeClustering(n_clusters=k).fit(X)
        re_kMeans = computeARI(kmeans.labels_, labels_true, X=X)
        re_Hierarchical = computeARI(hierarchical.labels_, labels_true, X=X)
        re = np.vstack((re_kMeans, re_Hierarchical))
        re_max = np.max(re, axis=0)
        print re_max
        for i in range(4, 100, 5):
            cblp = CBLP(model='uneven', topN=i).fit(X)
            print i,computeARI(cblp.labels_, labels_true, X=X)
            print i, (re_max < computeARI(cblp.labels_, labels_true, X=X)) * 1

    # for filename in evenFileList:
    #     X, labels_true = loadData(filename)
    #     k = len(set(labels_true))
    #     kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    #     hierarchical = AgglomerativeClustering(n_clusters=k).fit(X)
    #     re_kMeans = computeARI(kmeans.labels_, labels_true, X=X)
    #     re_Hierarchical = computeARI(hierarchical.labels_, labels_true, X=X)
    #     re = np.vstack((re_kMeans, re_Hierarchical))
    #     re_max = np.max(re, axis=0)
    #     print re_max
    #     for i in range(2,30,2):
    #         i=i*0.01
    #         cblp = CBLP(model='even', percent=i).fit(X)
    #         # print i,(re_max<computeARI(cblp.labels_, labels_true, X=X))*1
    #         print i,computeARI(cblp.labels_, labels_true, X=X)