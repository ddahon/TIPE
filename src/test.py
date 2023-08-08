import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
G = nx.Graph()
G.add_nodes_from([0,1,2,3])
G.add_weighted_edges_from([(1,0,0.5), (2,0,0.3), (3,0,0.2), (1, 2, 0.4)])
pos = nx.layout.spring_layout(G)
labels = dict()
##
couleurs = ['b', 'r', 'b', 'r']
labels[(1,0)] = 0.5
labels[(2,0)] = 0.1
labels[(3,0)] = 0.5
labels[(1,2)] = 0.4
nodelabels = dict()
nodelabels[0] = 0.7
nodelabels[1] = 0.9
nodelabels[2] = 0.5
nodelabels[3] = 0.3

nx.draw(G, pos, node_color = couleurs, alpha = 0.5, node_size = 1000)
nx.draw_networkx_edge_labels(G, pos, labels)
#nx.draw_networkx_labels(G, pos, nodelabels)
plt.show()

##
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes(projection='3d')
x = np.linspace(-100, 100, 10)
y = np.linspace(-100, 100, 10)
z = 2*x**2 - 3*y*
plt.show()
