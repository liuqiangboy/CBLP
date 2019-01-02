# -*- coding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import arrayOperation

NODE_SHAPE=['o','^','>','v','<','d','p','h','8','s','x','1','2','3','4']
COLOR=['white','skyblue','red','greenyellow','blue','black','pink','peru',
       'green','violet','tan']
# 画出列表中的节点
def drawNxNodes(propertysMatrix,node_list=None,node_color='w',withId=False):

    shape=np.shape(propertysMatrix)
    idList=np.arange(shape[0])

    if node_list==None:
        node_list=list(idList)
    if shape[1]!=2:
        xy=PCAchange(propertysMatrix,2)
    else:
        xy = propertysMatrix
    # xy=xy*5
    id_xy = dict(zip(idList, xy))
    id_l=dict(zip(idList, idList))
    G = nx.Graph()
    G.add_nodes_from(idList)
    # nx.draw_networkx_nodes(G, pos=id_xy,nodelist=node_list, node_size=20, node_color=node_color, edgecolors='w')
    # node_list=np.random.choice(idList,10)
    # id_l = dict(zip(node_list,node_list))
    if withId:
        nx.draw_networkx_labels(G, pos=id_xy,node_list=node_list, labels=id_l, font_size=10, font_family='sans-serif')
    return G
## 根据坐标绘制绘制点
def drawNxNodes_label(idList,xList,yList,lList,withLabel=False,node_size=20,withId=False):
    LABEL = list(set(lList))

    LABEL=sorted(LABEL)
    xy = list(zip(xList, yList))
    id_xy = dict(zip(idList, xy))

    id_l = dict(zip(idList, lList))
    id_id=dict(zip(idList, idList))
    # print ('id_xy:', id_xy)
    G = nx.Graph()
    G.add_nodes_from(idList)
    for k in id_l.keys():
        G.node[k]['label'] = id_l[k]
    # print ('G.nodes:', G.nodes(data=True))
    node_list = []
    for i in LABEL:
        node_list.append([u for u, d in G.nodes(data=True) if d['label'] == i])

    # draw nodes


    for i in range(len(LABEL)):
        nx.draw_networkx_nodes(G, pos=id_xy, nodelist=node_list[i], node_size=node_size, node_color=COLOR[i%len(COLOR)], edgecolors='black',
                               node_shape=NODE_SHAPE[i%(len(NODE_SHAPE))],label=i)
        plt.legend(ncol=3)
        # if withLabel:
            # plt.plot(x='top',label='test')

            # nx.draw_networkx_labels(G, pos=id_xy, labels=id_l, font_size=20, font_family='sans-serif')
        if withId:
            nx.draw_networkx_labels(G, pos=id_xy, labels=id_id, font_size=15, font_family='sans-serif')

    # plt.axis('off')
    # plt.show()
    return G

## 根据相似度绘制绘制边
def drawNxEdges(G,A,id_xy,with_weight=False):
    # print id_xy
    for i in range(np.shape(A)[0] - 1):
        for j in range(i + 1, np.shape(A)[0]):
            if A[i][j]!=0:
                G.add_edge(i, j, weight='%.2f' % (A[i][j]))
    # draw edges
    nx.draw_networkx_edges(G, pos=id_xy, width=0.5)
    if with_weight:
        # 获取边权重
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos=id_xy, edge_labels=edge_labels, font_size=3)
    return G

def drawTable(A):
    df=DataFrame(A)
    vals = np.around(df.values, 2)
    fig = plt.figure()
    ax = fig.add_subplot(111, frameon=True, xticks=[], yticks=[])
    the_table = plt.table(cellText=vals, rowLabels=df.index, colLabels=df.columns,
                          colWidths=[0.01] * vals.shape[1] ,loc='center', cellLoc='center')
    the_table.set_fontsize(20)

    the_table.scale(2.5, 2.58)
    plt.show()
def printDf(A):
    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)
    # 设置value的显示长度，默认为50
    # pd.set_option('max_colwidth', 30)
    print pd.DataFrame(np.round(A,2))
    # 控制索引
    # print pd.DataFrame(np.round(A,2),index=np.arange(1,np.shape(A)[0]+1),
    #                    columns=np.arange(1, np.shape(A)[1] + 1))

def drawA_Pdf(A,bins=10,name=''):
    data = A.flatten().tolist()
    plt.hist(data,bins=bins)
    plt.title(name+' PDF')
    plt.show()
def drawA_Cpf(A,bins=10,name=''):
    data = A.flatten().tolist()
    plt.hist(data, 100, cumulative=True, alpha=0.75)
    plt.title(name+' CPF')
    plt.show()
def drawSeedD(seedList,x_y):
    n = np.shape(x_y)[0]
    idList=list(np.arange(n))
    no_seedList = list(set(idList) - set(seedList))
    xList,yList=list(x_y[:,0]),list(x_y[:,1])
    drawNxNodes_color(idList, xList, yList, node_list=no_seedList)
    drawNxNodes_color(idList, xList, yList, node_list=seedList, node_color='r')
    plt.title('seed set')
    plt.show()
def drawLabel(propertysMatrix,LabelA,t=''):
    x_y=propertysMatrix
    if np.shape(x_y)[1]!=2:
        x_y=arrayOperation.PCAchange(x_y, n_components=2)
    n=np.shape(x_y)[0]
    if len(np.shape(LabelA))==1:
        lList=list(LabelA)
    else:
        k=np.shape(LabelA)[1]
        temp=np.sum(LabelA*np.arange(k),axis=1)
        temp[np.where(np.sum(LabelA,axis=1)==0)]=-1
        lList=list(map(int, temp))
    # ==============标签向量矩阵转为n*1的标签列表=================
    idList, xList, yList = list(np.arange(n)), x_y[:, 0], x_y[:, 1]
    drawNxNodes_label(idList, xList, yList, lList, node_size=20, withId=False,withLabel=False)
    # plt.title(t+',cluster num:'+str(len(set(lList))))
    plt.title(t)
    plt.axis('off')
    plt.show()
def drawNetwork(propertysMatrix,WA,lList=None,node_size=20,title="",withId=False):
    x_y = propertysMatrix
    if np.shape(propertysMatrix)[1] != 2:
        x_y =arrayOperation.PCAchange(propertysMatrix, n_components=2)
    n = np.shape(x_y)[0]
    idList, xList, yList = list(np.arange(n)), x_y[:, 0], x_y[:, 1]
    # 得到idList, xList, yList
    G=nx.Graph()
    if lList!=None and len(lList)!=0:
        G=drawNxNodes_label(idList,xList,yList,lList,node_size=node_size,withId=withId,withLabel=False)
    else:
        G = drawNxNodes_label(idList, xList, yList, list(np.zeros(n)), node_size=node_size, withId=withId, withLabel=False)
    G=drawNxEdges(G,WA,dict(zip(idList,list(x_y))),with_weight=False)
    plt.title(title)
    plt.axis('off')
    plt.show()
def drawHeatmap(A):
    sns.heatmap(A, annot=False, vmax=1, square=True, cmap="Blues")
    plt.show()

