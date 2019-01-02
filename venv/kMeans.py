# -*- coding: utf-8 -*-
from sklearn.cluster import KMeans
import numpy as np
from fileOperation import *
from arrayOperation import computeARI
from drawImage import *
from time import clock
def runKMeans(fileName,k=2):
    propertysMatrix, labelMatrix = loadData(fileName)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(propertysMatrix)
    myLabel = kmeans.labels_
    print computeARI(myLabel, labelMatrix)
    # drawLabel(propertysMatrix, myLabel, t='kmeans')
if __name__=='__main__':
    start=clock()
    # runKMeans(fileName='dataset/' + fileList['aggre'],k=7)
    # runKMeans(fileName='dataset/' + fileList['cancer'], k=2)
    # runKMeans(fileName='dataset/' + fileList['iris'],k=3)
    # runKMeans(fileName='dataset/' + fileList['seeds'], k=3)
    # runKMeans(fileName='dataset/' + fileList['wine'],k=3)
    # runKMeans(fileName='dataset/' + fileList['aw'],k=2)
    # runKMeans(fileName='dataset/' + fileList['cn'],k=2)
    # runKMeans(fileName='dataset/' + fileList['qz'],k=2)
    runKMeans(fileName='/Users/liuqiang/Documents/标签传播过程论文/聚类数据集/UCI/ionosphere.txt',k=2)
    end = clock()
    print('time cost:', end - start)



