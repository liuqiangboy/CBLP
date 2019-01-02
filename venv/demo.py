# -*- coding:utf-8 -*-
import networkx as nx

# 图像的显示：
#需要导入matplotlib
import matplotlib.pyplot as plt


G = nx.Graph()#创建空的网络图
'''networkx有四种图 Graph 、DiGraph、MultiGraph、MultiDiGraph，
分别为无多重边无向图、无多重边有向图、有多重边无向图、有多重边有向图。'''


# 加点和边
G.add_node('a')#添加点a

G.add_edge('x','y')#添加边,起点为x，终点为y
G.add_weighted_edges_from([('x','y',1.0)])#第三个输入量为权值
#也可以
list = [('a','b',5.0),('b','c',3.0),('a','c',1.0)]
G.add_weighted_edges_from([(list)])




nx.draw(G,pos = nx.random_layout(G),node_color = 'b',edge_color = 'r',with_labels = True,font_size =18,node_size =20)
plt.show()