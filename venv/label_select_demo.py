# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import networkx as nx

G = nx.Graph()
node_list=('a','b','c','d','e','f','g')
G.add_nodes_from(node_list)

pos = nx.spring_layout(G) # positions for all nodes
print pos
pos['a']=[0 , 0]
pos['b']=[0.58099828, 0.83849917]
pos['c']=[0.82413299, 0.12112204]
pos['f']=[-0.85769513,  0.25157874]
pos['d']=[ 0.69652735, -0.66052533]
pos['g']=[-0.73438295, -0.53082061]
pos['e']=[-0.41174914,  0.98014599]
# draw nodes
nx.draw_networkx_nodes(G, pos, node_size=2000,node_color='w',edgecolors='black',node_shape='o')
# draw node_label
node_label={'a':'2','b':'1','c':'1','d':'1','e':'3','f':'2','g':'2'}
node_with_label={}
for i in node_label:
    node_with_label.update({i:i+'('+node_label.get(i)+')'})
nx.draw_networkx_labels(G, pos,labels=node_with_label, font_size=20, font_family='sans-serif')
# draw edges
G.add_edge('a', 'b', weight=0.1)
G.add_edge('a', 'c', weight=0.1)
G.add_edge('a', 'd', weight=0.2)
G.add_edge('a', 'e', weight=0.5)
G.add_edge('a', 'f', weight=0.3)
G.add_edge('a', 'g', weight=0.3,)
nx.draw_networkx_edges(G, pos,width=6)
# draw edges weights labels
wDcit={}
for (u, v, d) in G.edges(data=True):
    d={(u,v):d['weight']}
    wDcit.update(d)

nx.draw_networkx_edge_labels(G,pos,wDcit,font_size=20)
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