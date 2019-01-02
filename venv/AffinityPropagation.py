from sklearn.cluster import AffinityPropagation
import numpy as np
from fileOperation import loadData,fileList
from arrayOperation import computeARI
from drawImage import drawLabel
from time import clock
def runAff(fileName,title='AffinityPropagation'):
    propertysMatrix, labelMatrix = loadData(fileName)
    cluster = AffinityPropagation().fit(propertysMatrix)
    myLabel = cluster.labels_
    re= computeARI(myLabel, labelMatrix)
    print re
    drawLabel(propertysMatrix, myLabel, t=title)

if __name__=='__main__':
    start=clock()
    runAff(fileName='dataset/' + fileList['aggre'])
    runAff(fileName='dataset/' + fileList['cancer'])
    runAff(fileName='dataset/' + fileList['iris'])
    runAff(fileName='dataset/' + fileList['seeds'])
    runAff(fileName='dataset/' + fileList['wine'])
    runAff(fileName='dataset/' + fileList['aw'])
    runAff(fileName='dataset/' + fileList['qz'])
    runAff(fileName='dataset/' + fileList['qz'])
    end=clock()
    print('time cost:',end-start)