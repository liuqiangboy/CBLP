# -*- coding: utf-8 -*-
from flow import *
import os
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator
if __name__ =='__main__':
    fileList={
        # '/Users/liuqiang/Documents/标签传播过程论文/聚类数据集/uneven/heart.txt':'uneven',
        # '/Users/liuqiang/Documents/标签传播过程论文/聚类数据集/uneven/Cancer.txt': 'uneven',
        #
        # '/Users/liuqiang/Documents/标签传播过程论文/聚类数据集/uneven/vote.txt':'uneven',
        # '/Users/liuqiang/Documents/标签传播过程论文/聚类数据集/uneven/iris.txt':'uneven',
        '/Users/liuqiang/Documents/标签传播过程论文/聚类数据集/even/five_cluster.txt': 'even',
        '/Users/liuqiang/Documents/标签传播过程论文/聚类数据集/uneven/wine.txt': 'uneven',
        # '/Users/liuqiang/Documents/标签传播过程论文/聚类数据集/even/letterqz.txt':'even',
        # '/Users/liuqiang/Documents/标签传播过程论文/聚类数据集/even/letteraw.txt':'even',
        # '/Users/liuqiang/Documents/标签传播过程论文/聚类数据集/even/seeds.txt':'even',
        # '/Users/liuqiang/Documents/标签传播过程论文/聚类数据集/even/Aggregation.txt':'even',

    }

    evenFileList=[]
    evenPath='/Users/liuqiang/Documents/标签传播过程论文/聚类数据集/even/'
    for i in os.listdir(evenPath):
        if i.split('.')[-1] == 'txt':
            evenFileList.append(evenPath+i)
    unevenFileList = []
    unevenPath = '/Users/liuqiang/Documents/标签传播过程论文/聚类数据集/uneven/'
    for i in os.listdir(unevenPath):
        if i.split('.')[-1] == 'txt':
            unevenFileList.append(unevenPath + i)

    marker_shape =['o','*','^','d','s']
    line_style=['-','--','-.',':']
    marker_index = 0
    line_index=0
    x_min=0.00
    x_max=0.15
    scale=0.01
    evenFileList, unevenFileList = [],[]
    for filename in fileList.keys():
        if fileList[filename]=='even':
            evenFileList.append(filename)
        elif fileList[filename]=='uneven':
            unevenFileList.append(filename)
        else:
            print 'some error appear'
            break
    for filename in evenFileList+unevenFileList:
        SIA=file2SIA(filename)
        x,y=[],[]
        marker = marker_shape[marker_index]
        line=line_style[line_index]
        marker_index = (marker_index + 1 )% len(marker_shape)
        line_index=(line_index+1) % len(line_style)
        for percent in np.arange(x_min,x_max+scale,scale):
            if percent == 0.05:
                print filename,computeNoise(SIA2WA(SIA,percent=percent))
            if percent>=0.05 and int(percent*100)%2==0:
                continue
            x.append(percent)
            y.append(computeNoise(SIA2WA(SIA,percent=percent)))

        # plt.plot(x, y, mec='r', mfc='w', marker=marker, linestyle=line,label=filename.split('/')[-1].split('.')[0])
        if filename in evenFileList:
            plt.plot(x, y, '--', mec='red', mfc='w',color='red', marker=marker,MarkerSize=7,
                     label=filename.split('/')[-1].split('.')[0])
        else:
            plt.plot(x, y, ':',mec='blue', mfc='w', color='blue', marker=marker,MarkerSize=7,
                     label=filename.split('/')[-1].split('.')[0])
        plt.legend(ncol=2)
    # plt.plot([x_min,x_max],[0.0025,0.0025])
    plt.xlabel('keep percent')
    plt.ylabel('noise percent')
    # plt.plot([x_min,x_max],[0.05,0.05])
    plt.yticks(np.hstack((np.arange(0.,0.1,step=0.05),np.array([0.1,0.2,0.3,0.5,1]))))
    # plt.yticks(np.hstack((np.arange(0.,0.1,step=0.01))))
    plt.xticks(np.hstack((np.arange(0,0.05,step=0.01),np.arange(0.05,0.17,step=0.02))))
    plt.grid(axis='y',linestyle=':')
    plt.grid(axis='x',linestyle=':')
    plt.show()