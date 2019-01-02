# -*- coding: utf-8 -*-
import numpy as np
import os
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans

from CBLP import CBLP
from arrayOperation import computeARI,computeNoise,SIA2WA
from fileOperation import loadData,file2SIA
from drawImage import drawLabel
if __name__ =='__main__':
    fileList = {
        '/Users/liuqiang/Documents/标签传播过程论文/聚类数据集/uneven/heart.txt': 'uneven',
        '/Users/liuqiang/Documents/标签传播过程论文/聚类数据集/uneven/cancer.txt': 'uneven',
        '/Users/liuqiang/Documents/标签传播过程论文/聚类数据集/uneven/wine.txt': 'uneven',
        '/Users/liuqiang/Documents/标签传播过程论文/聚类数据集/uneven/vote.txt': 'uneven',
        '/Users/liuqiang/Documents/标签传播过程论文/聚类数据集/even/letterqz.txt': 'even',
        '/Users/liuqiang/Documents/标签传播过程论文/聚类数据集/even/letteraw.txt': 'even',
        '/Users/liuqiang/Documents/标签传播过程论文/聚类数据集/even/seeds.txt': 'even',
        '/Users/liuqiang/Documents/标签传播过程论文/聚类数据集/even/aggregation.txt':'even',

        # '/Users/liuqiang/Documents/标签传播过程论文/聚类数据集/uneven/iris.txt': 'uneven',
        # '/Users/liuqiang/Documents/标签传播过程论文/聚类数据集/even/five_cluster.txt':'even',
    }
    # runDBSCAN(fileName='dataset/' + fileList['aggre'],eps=0.1,min_samples=36)
    # runDBSCAN(fileName='dataset/' + fileList['cancer'],eps=0.5,min_samples=40)
    # runDBSCAN(fileName='dataset/' + fileList['iris'],eps=0.4,min_samples=2)
    # runDBSCAN(fileName='dataset/' + fileList['seeds'],eps=0.3,min_samples=21)
    # runDBSCAN(fileName='dataset/' + fileList['wine'],eps=0.6,min_samples=37)

    # runDBSCAN(fileName='dataset/' + fileList['aw'],eps=0.7,min_samples=14)
    # runDBSCAN(fileName='dataset/' + fileList['cn'],eps=0.6,min_samples=27)
    # runDBSCAN(fileName='dataset/' + fileList['qz'],eps=0.4,min_samples=4)
    quality=['ARI','AMI','V-measure','FMI','NMI']
    DBSCAN_para={
        'heart':[0.3,2],
        'cancer':[0.5,40],
        'wine':[0.6,37],
        'vote':[0.9,10],
        'letterqz':[0.4,4],
        'letteraw':[0.7,14],
        'seeds':[0.3,21],
        'aggregation':[0.1,36],
        'five_cluster':[0.1,36],

    }

    result=[]
    data=[]
    for filename in fileList.keys():
        X, labels_true = loadData(filename)
        # drawLabel(X, LabelA=labels_true, )
        # 原始聚类
        # 其他算法最好的结果
        k = len(set(labels_true))
        ap = AffinityPropagation().fit(X)
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        hierarchical = AgglomerativeClustering(n_clusters=k).fit(X)
        f=filename.split('/')[-1].split('.')[0]
        dbscan=DBSCAN(eps=DBSCAN_para[f][0], min_samples=DBSCAN_para[f][1],metric='euclidean').fit(X)

        re_dbscan=computeARI(dbscan.labels_, labels_true, X=X)
        re_ap=computeARI(ap.labels_, labels_true, X=X)
        re_kMeans = computeARI(kmeans.labels_, labels_true, X=X)
        re_Hierarchical = computeARI(hierarchical.labels_, labels_true, X=X)
        re_max = np.max(np.vstack((re_kMeans, re_Hierarchical,re_ap,re_dbscan)), axis=0)

        # ===============
        noisePercent=computeNoise(SIA2WA(file2SIA(filename), percent=0.05))
        if noisePercent<0.05:
            percent=2*noisePercent+0.1
            cblp = CBLP(model='even',percent=percent).fit(X)
            re_cblp=computeARI(cblp.labels_, labels_true, X=X)
            print ('model:even,percent:',percent,re_cblp,(re_max < computeARI(cblp.labels_, labels_true, X=X)) * 1)
        else:
            topN=int(35.*np.log(np.shape(X)[0])-145)+1
            cblp = CBLP(model='uneven', topN=topN).fit(X)
            re_cblp=computeARI(cblp.labels_, labels_true, X=X)
            print ('model:uneven,topN:',topN, re_cblp, (re_max < computeARI(cblp.labels_, labels_true, X=X)) * 1)
        result.append(re_ap)
        result.append(re_dbscan)
        result.append(re_kMeans)
        result.append(re_Hierarchical)
        result.append(re_cblp)
        data.append(f)

    algorithm_num=5
    a = np.zeros(shape=(algorithm_num,len(fileList)))
    # 4个算法，8个数据集
    result=np.array(result)
    for q in range(len(quality)):
        for i in range(0,algorithm_num*len(fileList),algorithm_num):
    #         0-32个值，32=4个算法*8个数据集
            a[0:algorithm_num, i / algorithm_num] = result[i:i + algorithm_num, q]
        np.savetxt('quality/'+quality[q]+'.txt', a.transpose(), fmt='%.4f')
    np.savetxt('quality/reanme.txt',[],header='ap dbscan kmeans hierarchical cblp'+str(data))


    # runCBLP(fileName='dataset/' + fileList['aggre'],percent=0.1)
    # runCBLP(fileName='dataset/' + fileList['aggre'],percent=0.05)
    # runCBLP(fileName='dataset/' + fileList['cancer'], topN=30, WA2SIAway='adapt')
    # runCBLP(fileName='dataset/' + fileList['iris'], WA2SIAway='adapt', topN=30)
    # runCBLP(fileName='dataset/' + fileList['seeds'], WA2SIAway='adapt', topN=30,)
    # runCBLP(fileName='dataset/' + fileList['seeds'], percent=0.15,)

    # runCBLP(fileName='dataset/' + fileList['wine'],WA2SIAway='adapt', topN=30,)
    # runCBLP(fileName='dataset/' + fileList['aw'], percent=0.1)
    # runCBLP(fileName='dataset/' + fileList['aw'], percent=0.15)
    # runCBLP(fileName='dataset/' + fileList['cn'], percent=0.1)
    # runCBLP(fileName='dataset/' + fileList['cn'], percent=0.1125)

    # runCBLP(fileName='dataset/' + fileList['qz'], percent=0.1)
    # runCBLP(fileName='dataset/' + fileList['qz'],  WA2SIAway='adapt', topN=50)
    # runCBLP(fileName='dataset/' + fileList['three'], WA2SIAway='adapt', topN=30, percent=0.1)
    # runCBLP(fileName='dataset/' + fileList['five'], WA2SIAway='adapt', topN=30,percent=0.1)

    # runCBLP(fileName='dataset/' + fileList['jain'], WA2SIAway='adapt', topN=30, )
    # runCBLP(fileName='dataset/' + fileList['moons'], WA2SIAway='adapt', topN=30, )
    # runCBLP(fileName='dataset/' + fileList['sym'], WA2SIAway='adapt', topN=30, )

    # runCBLP(fileName='dataset/' + fileList['WDBC'], WA2SIAway='adapt', topN=40, )
    # runCBLP('/Users/liuqiang/PycharmProjects/CBLP/venv/dataset/UCI/landsat.txt', percent=0.15 )
    # runCBLP('/Users/liuqiang/PycharmProjects/CBLP/venv/dataset/UCI/landsat.txt', percent=0.1 )

    # runCBLP(fileName='dataset/' + fileList['sym'],
    #         iterationTime=30, percent=0.1)
    # runCBLP(fileName='dataset/' + fileList['moons'],
    #         iterationTime=30, percent=0.1)
    # runCBLP(fileName='dataset/' + fileList['jain'],
    #         iterationTime=30, percent=0.1)
    # 合并阈值20  有两个社区未合并


    #论文 3 画正确类图
    # fileName='dataset/' + fileList['cancer']
    #
    # propertysMatrix, labelMatrix = loadData(fileName)
    # n=np.shape(propertysMatrix)
    # drawLabel(propertysMatrix, LabelA=labelMatrix,
    #           t=fileName.split('/')[-1].split('.')[0])
    #
    percent=0.05
    # print(computeNoise(SIA2WA(file2SIA('dataset/' + fileList['three']),percent=percent)))
    # print(computeNoise(SIA2WA(file2SIA('dataset/' + fileList['five']),percent=percent)))
    # print(computeNoise(SIA2WA(file2SIA('dataset/' + fileList['cancer']),percent=percent)))
    # print(computeNoise(SIA2WA(file2SIA('dataset/' + fileList['iris']), percent=percent)))
    # print(computeNoise(SIA2WA(file2SIA('dataset/' + fileList['seeds']), percent=percent)))
    # print(computeNoise(SIA2WA(file2SIA('dataset/' + fileList['wine']), percent=percent)))
    # print(computeNoise(SIA2WA(file2SIA('dataset/' + fileList['aggre']), percent=percent)))
    # print(computeNoise(SIA2WA(file2SIA('dataset/' + fileList['aw']), percent=percent)))
    # print(computeNoise(SIA2WA(file2SIA('dataset/' + fileList['qz']), percent=percent)))
    # print(computeNoise(SIA2WA(file2SIA('dataset/' + fileList['cn']), percent=percent)))
