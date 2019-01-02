# -*- coding: utf-8 -*-
from CBLP import CBLP
from fileOperation import loadData
from arrayOperation import computeARI
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import warnings
warnings.filterwarnings('ignore')
if __name__=='__main__':
    X,labels_true=loadData('/Users/liuqiang/PycharmProjects/CBLP/venv/dataset/UCI/landsat.txt')
    k=len(set(labels_true))

    # dbscan = DBSCAN(eps=eps, min_samples=min_samples,metric='euclidean').fit(X)
    kmeans= KMeans(n_clusters=k, random_state=0).fit(X)
    hierarchical = AgglomerativeClustering(n_clusters=k).fit(X)
    # cblp = CBLP(model='even', percent=0.1).fit(X)
    cblp = CBLP(model='uneven', topN=20).fit(X)
    print 'CBLP:',computeARI(cblp.labels_, labels_true, X=X)
    print 'k-Means:', computeARI(kmeans.labels_, labels_true, X=X)
    print 'Hierarchical:', computeARI(hierarchical.labels_, labels_true, X=X)

