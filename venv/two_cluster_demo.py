# -*- coding: utf-8 -*-
import networkx as nx
import matplotlib.pyplot as plt
import pylab

NODE_SHAPE=['o','^','>','v','<','d','p','h','8','s']
## 从.txt文件中读取数据
def loadData(flieName):
    inFile = open(flieName, 'r')  # 以只读方式打开某fileName文件

    # 定义两个空list，用来存放文件中的数据
    L = []
    X = []
    Y = []
    i=1
    ID=[]
    for line in inFile:
        trainingSet = line.split(' ')  # 对于每一行，按','把数据分开，这里是分成两部分
        ID.append(i)
        i=i+1
        L.append(int(trainingSet[0].split('.')[0]))  # 标签部分，即文件中的第一列数据逐一添加到list L 中
        X.append(float(trainingSet[1]))  # 属性第一部分，即文件中的第二列数据逐一添加到list X 中
        Y.append(float(trainingSet[2].strip('\n')))  # 属性第二部分，即文件中的第三列数据逐一添加到list y 中

    return (ID, X, Y,L)  # X,Y组成一个元组，这样可以通过函数一次性返回
## 根据坐标绘制绘制点
def plotData(X, y):
    length = len(y)

    pylab.figure(1)

    pylab.plot(X, y, 'rx')
    pylab.xlabel('Population of City in 10,000s')
    pylab.ylabel('Profit in $10,000s')

    pylab.show()  # 让绘制的图像在屏幕上显示出来


idList,xList,yList,lList=loadData('two_cluster.txt')
LABEL=list(set(lList))
xy=list(zip(xList,yList))
id_xy=dict(zip(idList,xy))
id_l=dict(zip(idList,lList))
print ('id_xy:',id_xy)
G = nx.Graph()
G.add_nodes_from(idList)
for k in id_l.keys():
    G.node[k]['label']=id_l[k]
print ('G.nodes:',G.nodes(data=True))
node_list=[]
for i in LABEL:
    node_list.append([u for u,d in G.nodes(data=True) if d['label'] == i])

# draw nodes
for i in range(len(LABEL)):
    nx.draw_networkx_nodes(G, pos=id_xy,nodelist=node_list[i],
                           node_size=20,node_color='w',edgecolors='black',node_shape=NODE_SHAPE[i])

plt.axis('off')
plt.show()
