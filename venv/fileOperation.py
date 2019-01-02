# -*- coding: utf-8 -*-
from arrayOperation import *
import numpy as np
import pandas as pd
fileList={'two':'artificial/two_cluster.txt','two_10':'artificial/two_cluster_10.txt','two_20':'artificial/two_cluster_20.txt',
          'three':'artificial/three_cluster.txt','five':'artificial/five_cluster.txt','spiral':'artificial/spiral.txt',
          'spiral_':'artificial/spiral_unbalance.txt',
'aggre':'UCI/Aggregation.txt',
'aw':'UCI/letteraw.txt',
'cn':'UCI/lettercn.txt',
'qz':'UCI/letterqz.txt',

'cancer':'UCI/Cancer.txt',
'iris':'UCI/iris.txt',
'seeds':'UCI/seeds.txt',
'wine':'UCI/wine.txt',
'thyroid':'UCI/Thyroid.txt',


'jain':'UCI/Jain.txt',
'moons':'artificial/Twomoons.txt',
'sym':'UCI/sym.txt',
'WDBC':'UCI/WDBC.txt',

'temp':'artificial/D31.txt',
          }
UCIpath='/Users/liuqiang/Documents/标签传播过程论文/聚类数据集/UCI/'

## 从.txt文件中读取数据,返回N*P的属性矩阵，N*1的标签矩阵
def loadData(fileName):
    matrix=read_file2matrix(fileName)
    propertysMatrix=matrix[:,1:]
    if np.shape(propertysMatrix)[1]>20:
        propertysMatrix=PCAchange(propertysMatrix,2)
        print 'PCA change'
    colMin = np.min(propertysMatrix, axis=0)
    colMax = np.max(propertysMatrix, axis=0)
    propertysMatrix = (propertysMatrix - colMin) / (colMax - colMin)
    # # 数据归一化
    labelMatrix=matrix[:,0].astype(np.int8)
    print("load :",fileName.split('/')[-1],' get data:',np.shape(propertysMatrix)[0])
    return (propertysMatrix,labelMatrix)

## 从.txt文件中读取数据
def read_file2matrix(fileName):
    #首先读取文件
    matrix = np.loadtxt(fileName,delimiter=' ')
    # print(matrix.shape)
    return matrix
# 将txt文件改为标签在第一列的格式
def file2SIA(fileName):
    matrix = read_file2matrix(fileName)
    propertysMatrix = matrix[:, 1:]
    if np.shape(propertysMatrix)[1]>20:
        propertysMatrix=PCAchange(propertysMatrix,15)
    # colSum=np.sum(propertysMatrix,axis=0)
    # zeroColIndex=np.where(colSum==0)
    # propertysMatrix=np.delete(propertysMatrix,zeroColIndex,axis=1)
    # # 去掉全为零的列
    # colMin = np.min(propertysMatrix, axis=0)
    # colMax = np.max(propertysMatrix, axis=0)
    # propertysMatrix=propertysMatrix/colMax
    # propertysMatrix = (propertysMatrix - colMin) / (colMax - colMin)
    # # 数据归一化
    # n = np.shape(propertysMatrix)[0]
    # ====================导入数据集=========================
    EDA = squareform(pdist(propertysMatrix, metric='seuclidean'))
    # ============计算标准欧式距离============================
    SIA = EDA2SIA(EDA)
    return SIA

def changeTxt(path,fileName):
    matrix = np.loadtxt(path+fileName,delimiter=',')
    # diabetes = pd.read_csv('dataset/UCI/diabetes.csv')
    # matrix = np.array(diabetes.values)
    # print np.shape(matrix)
    matrix[:,[0,2]]=matrix[:,[2,0]]
    # matrix=np.around(matrix,decimals=4)
    # for i in range(26):
    #     matrix1=matrix[np.where(matrix[:,0]==(i+1)*1.0)]
    np.savetxt(path+'temp.txt',matrix,fmt='%.4f')

def main():
    m = np.loadtxt('result.txt', delimiter=' ')
if __name__ =='__main__':
    main()