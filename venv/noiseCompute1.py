# -*- coding: utf-8 -*-
from flow import *
import os
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator
if __name__ =='__main__':
    filePath='/Users/liuqiang/Documents/标签传播过程论文/聚类数据集/test/'
    fileList=[]
    for i in os.listdir(filePath):
        if i.split('.')[-1] == 'txt':
            fileList.append(filePath+i)
    # fileList = ['/Users/liuqiang/PycharmProjects/CBLP/venv/dataset/UCI/liver.txt']
    x_min = 0.00
    x_max = 0.15
    scale = 0.01
    for filename in fileList:
        SIA=file2SIA(filename)
        x,y=[],[]
        noise_5=computeNoise(SIA2WA(SIA,percent=0.05))
        for percent in np.arange(x_min,x_max+scale,scale):
            x.append(percent)
            y.append(computeNoise(SIA2WA(SIA,percent=percent)))

        plt.plot(x, y, '-.', mfc='w',
                     label=filename.split('/')[-1].split('.')[0])

        plt.legend(ncol=3)
    # plt.plot([x_min,x_max],[0.0025,0.0025])
    plt.xlabel('keep percent')
    plt.ylabel('noise percent')
    # plt.yticks(np.array([0.05]))
    # plt.xticks(np.hstack((np.array([0.05]))))
    # plt.grid(axis='y',linestyle=':')
    # plt.grid(axis='x',linestyle=':')
    plt.show()