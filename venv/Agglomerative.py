from sklearn.cluster import AgglomerativeClustering
import numpy as np
from fileOperation import *
from arrayOperation import computeARI
from drawImage import *
def runAgglomerative(fileName,k=2):
    propertysMatrix, labelMatrix = loadData(fileName)
    re= AgglomerativeClustering(n_clusters=k).fit(propertysMatrix)
    myLabel = re.labels_
    print computeARI(myLabel, labelMatrix)
    # drawLabel(propertysMatrix, myLabel, t='kmeans')
if __name__=='__main__':
    start=clock()
    # runAgglomerative(fileName='dataset/' + fileList['aggre'],k=7)
    # runAgglomerative(fileName='dataset/' + fileList['iris'],k=3)
    # runAgglomerative(fileName='dataset/' + fileList['cancer'],k=2)
    # runAgglomerative(fileName='dataset/' + fileList['wine'],k=3)
    # runAgglomerative(fileName='dataset/' + fileList['seeds'],k=3)
    # runAgglomerative(fileName='dataset/' + fileList['aw'],k=2)
    # runAgglomerative(fileName='dataset/' + fileList['cn'],k=2)
    # runAgglomerative(fileName='dataset/' + fileList['qz'],k=2)

    runAgglomerative(fileName='/Users/liuqiang/PycharmProjects/CBLP/venv/dataset/UCI/landsat.txt',k=6)


    end = clock()
    print('time cost:', end - start)



