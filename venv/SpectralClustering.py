from sklearn.cluster import SpectralClustering
import numpy as np
from fileOperation import *
from arrayOperation import computeARI
from drawImage import *
def runSpectralClustering(fileName,k=2):
    propertysMatrix, labelMatrix = loadData(fileName)
    re= SpectralClustering(n_clusters=k).fit(propertysMatrix)
    myLabel = re.labels_
    print computeARI(myLabel, labelMatrix)
    drawLabel(propertysMatrix, myLabel, t='SpectralClustering')
if __name__=='__main__':
    runSpectralClustering(fileName='dataset/' + fileList['aggre'],k=7)
    runSpectralClustering(fileName='dataset/' + fileList['iris'],k=3)
    runSpectralClustering(fileName='dataset/' + fileList['cancer'],k=2)
    runSpectralClustering(fileName='dataset/' + fileList['wine'],k=3)
    runSpectralClustering(fileName='dataset/' + fileList['seeds'],k=3)



