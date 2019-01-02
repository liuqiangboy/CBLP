# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import networkx as nx

G = nx.Graph()
node_list=['a','b','c','d','e','f','x','y','z']

G.add_nodes_from(node_list)
# G.add_nodes_from(invisible_node_list)
pos = nx.spring_layout(G) # positions for all nodes
print pos
pos['a']=[0 , 0]
pos['b']=[0.58099828, 0.83849917]
pos['c']=[0.82413299, 0.12112204]
pos['d']=[ 0.69652735, -0.66052533]
pos['e']=[-0.85769513,  0.24]
pos['f']=[-1.7,  0.24]

pos['x']=[1.5, 0.7]
pos['y']=[1.82413299, 0.12112204]
pos['z']=[ -0.39652735, -0.76052533]
# draw nodes
nx.draw_networkx_nodes(G, pos, node_size=2000,node_color='w',edgecolors='black',node_shape='o')
# draw node_label
node_label={'a':'1','b':'1','c':'1','d':'1','e':'2','f':'2','x':'1','y':'1','z':'1'}
node_with_label={}
for i in node_label:
    node_with_label.update({i:i+'('+node_label.get(i)+')'})
nx.draw_networkx_labels(G, pos,labels=node_with_label, font_size=20, font_family='sans-serif')
# draw edges
G.add_edge('a', 'b', weight=0.6)
G.add_edge('a', 'c', weight=0.8)
G.add_edge('a', 'd', weight=0.7)
G.add_edge('c', 'd', weight=0.7)
G.add_edge('a', 'e', weight=0.8)
G.add_edge('e', 'f', weight=0.85)

G.add_edge('b','x',weight=0.8)
G.add_edge('b','y',weight=0.65)
G.add_edge('c','y',weight=0.8)
G.add_edge('d','z',weight=0.8)
G.add_edge('d','y',weight=0.7)
nx.draw_networkx_edges(G, pos,width=4)
# draw edges weights labels
wDcit={}
for (u, v, d) in G.edges(data=True):
    d={(u,v):d['weight']}
    wDcit.update(d)

nx.draw_networkx_edge_labels(G,pos,wDcit,font_size=10)
plt.axis('off')
plt.show()


'''
elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.5]
esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.5]

# edges
nx.draw_networkx_edges(G, pos, edgelist=elarge,
                       width=6)
nx.draw_networkx_edges(G, pos, edgelist=esmall,
                       width=6, alpha=0.5, edge_color='b', style='dashed')
'''