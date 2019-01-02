# -*- coding: utf-8 -*-
from flow import *
import networkx as nx
import matplotlib.pyplot as plt
import pylab
from arrayOperation import *
import numpy as np
from sklearn.linear_model import LinearRegression
from fileOperation import *
# fileList={'two':'two_cluster.txt','two_10':'two_cluster_10.txt','two_20':'two_cluster_20.txt','three':'three_cluster.txt',
#           'five':'five_cluster.txt','spiral':'spiral.txt','spiral_':'spiral_unbalance.txt',
#           'moons':'Twomoons.txt','iris':'iris.txt'}
# score=np.array([[0,0],[0,0]])
# TPFP=com(2,449)+com(2,234)
# score[0,0]=com(2,433)+com(2,233)+com(2,16)+com(2,11)
# score[1,0]=TPFP-score[0,0]
#
# FNTN=com(1,433)*com(1,234)
# score[0,1]=com(1,233)*com(1,16)+com(1,433)*com(1,11)
# score[1,1]=FNTN-score[0,1]
# print score
# print 'cancer RI',(score[0,0]+score[1,1])*1.0/np.sum(score)
# # ===========cancer=========================================
# score=np.array([[0,0],[0,0]])
# TPFP=com(2,62)+com(2,50)+com(2,66)
# score[0,0]=com(2,59)+com(2,48)+com(2,66)+com(2,3)+com(2,2)
# score[1,0]=TPFP-score[0,0]
#
# FNTN=com(1,62)*com(1,50)+com(1,62)*com(1,66)+com(1,50)*com(1,66)
# score[0,1]=com(1,3)*com(1,66)+com(1,2)*com(1,66)
# score[1,1]=FNTN-score[0,1]
# print score
# print 'wine RI',(score[0,0]+score[1,1])*1.0/np.sum(score)
# =================wine===================================
