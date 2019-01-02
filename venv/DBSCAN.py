# -*- coding: utf-8 -*-
from sklearn.cluster import DBSCAN
import numpy as np
from fileOperation import loadData,fileList
from arrayOperation import computeARI
from drawImage import drawLabel
from time import clock
def runDBSCAN(fileName,eps,min_samples,title='DBSCAN'):
    propertysMatrix, labelMatrix = loadData(fileName)
    cluster = DBSCAN(eps=eps, min_samples=min_samples,metric='euclidean').fit(propertysMatrix)
    myLabel = cluster.labels_
    re= computeARI(myLabel, labelMatrix),eps,min_samples
    print re
    # drawLabel(propertysMatrix, myLabel, t=title)
    return re[0][-3]
if __name__=='__main__':
    start=clock()
    # runDBSCAN(fileName='dataset/' + fileList['aggre'],eps=0.1,min_samples=36)
    # runDBSCAN(fileName='/Users/liuqiang/PycharmProjects/CBLP/venv/dataset/UCI/landsat.txt',eps=0.4,min_samples=12)

    # runDBSCAN(fileName='dataset/' + fileList['cancer'],eps=0.5,min_samples=40)
    # runDBSCAN(fileName='dataset/' + fileList['iris'],eps=0.4,min_samples=2)
    # runDBSCAN(fileName='dataset/' + fileList['seeds'],eps=0.3,min_samples=21)
    # runDBSCAN(fileName='dataset/' + fileList['wine'],eps=0.6,min_samples=37)
    #
    # runDBSCAN(fileName='dataset/' + fileList['aw'],eps=0.7,min_samples=14)
    # runDBSCAN(fileName='dataset/' + fileList['cn'],eps=0.6,min_samples=27)
    # runDBSCAN(fileName='dataset/' + fileList['qz'],eps=0.4,min_samples=4)
    end = clock()
    print('time cost:', end - start)


    NMI_max=0
    eps_max=0
    min_e=0
    for eps in range(1,100,2):
        for min_samples in range(9,15):
            NMI=runDBSCAN(fileName='/Users/liuqiang/Documents/标签传播过程论文/聚类数据集/uneven/vote.txt', eps=eps*0.1,
                          min_samples=min_samples,title='DBSCAN')
            if NMI_max<NMI:
                NMI_max=NMI
                eps_max=eps
                min_e=min_samples
    print 'NMI_max:',NMI_max,eps_max,min_e