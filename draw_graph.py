import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt

plt.ion()
G=dnx.chimera_graph(16, 16, 4)
print(G.edges)