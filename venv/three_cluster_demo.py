# -*- coding: utf-8 -*-
import networkx as nx
import matplotlib.pyplot as plt
import pylab
import numpy as np
import seaborn as sns
import pandas as pd
import copy

from drawImage import *
from fileOperation import *
from arrayOperation import *

# idList,xList,yList,lList=loadData('artificial/five_cluster.txt')
idList,xList,yList,lList=loadData('artificial/three_cluster.txt')
# idList,xList,yList,lList=loadData('artificial/two_cluster_10.txt')
# idList,xList,yList,lList=loadData('artificial/spiral.txt')
# idList,xList,yList,lList=loadData('artificial/spiral_unbalance.txt')
# idList,xList,yList,lList=loadData('artificial/Twomoons.txt')
xy = list(zip(xList, yList))
id_xy = dict(zip(idList, xy))
xyArray=np.array(xy).T
# 距离矩阵到相似度矩阵的一些处理
EDA=np.sqrt(compute_squared_EDA(xyArray))
SIA=EDA2SIA(EDA)
# =====================================
# 计算连接矩阵LINKA (n,n)
LINKA=(filter(SIA,threshold=0.7)>0) * 1
# ==================
# 计算得到种子集合
seedList=climbAlgorithm(LINKA)
no_seedList=list(set(idList)-set(seedList))
drawNxNodes_color(idList,xList,yList,node_list=no_seedList)
drawNxNodes_color(idList,xList,yList,node_list=seedList,node_color='r')
plt.show()


# 画节点和边
# G=drawNxNodes(idList,xList,yList,lList,node_size=50,withId=False)
# G=drawNxEdges(G,filter(SIA),id_xy)
# plt.show()

# =========================================

# drawHeatmap(SIA)