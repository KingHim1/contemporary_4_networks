import networkx as nx
import random
import queue
from summativeQuestion1 import make_ring_graph

file = open("coauthorship.txt")
text = file.read()
text = text.split("*")

verticesFromText = text[1].split("\n")
verticesToName = {}
cite_graph = nx.DiGraph()
for i in range(1, len(verticesFromText)-1):
    verticesToName[i] = verticesFromText[i].split("\"")[1]
    cite_graph.add_node(i)

edgesFromText = text[2].split("\n")
for i in range(1, len(edgesFromText)-1):
    edges = edgesFromText[i].split(" ")
    # print(edges[2] + ": " + edges[3])
    if int(edges[2]) != int(edges[3]):
        cite_graph.add_edge(int(edges[2]), int(edges[3]))
#
xdata = []
ydata = []
cite_dist = {}
for vertex in cite_graph.adj:
    # xdata.append(vertex)
    if len(cite_graph.adj[vertex]) != 0:
        H = cite_graph.subgraph(cite_graph.adj[vertex]).to_undirected()
        # print(nx.maximal_independent_set(H))
        val = len(nx.maximal_independent_set(H))
        # ydata.append(val)
        if val in cite_dist:
            cite_dist[val] += 1
        else:
            cite_dist[val] = 1
    # else:
        # ydata.append(0)

ring_graph = nx.DiGraph()
H = make_ring_graph(50, 30, 0.4, 0.1)
for x in H:
    for y in H[x]:
        ring_graph.add_edge(int(x), int(y))

xdata1 = []
ydata1 = []
ring_dist = {}
for vertex in ring_graph.adj:
    # xdata1.append(vertex)
    if len(ring_graph.adj[vertex]) != 0:
        H = ring_graph.subgraph(ring_graph.adj[vertex]).to_undirected()
        # print(nx.maximal_independent_set(H))
        val = len(nx.maximal_independent_set(H))
        # ydata1.append(val)
        if val in ring_dist:
            ring_dist[val] +=1
        else:
            ring_dist[val] = 1
    # else:
    #     ydata1.append(0)

for x in ring_dist:
    xdata1.append(x)
    ydata1.append(ring_dist[x])

for y in cite_dist:
    xdata.append(y)
    ydata.append(cite_dist[y])



import matplotlib.pyplot as plt
plt.xlabel('node index')
plt.ylabel('Brilliance')
plt.title('Brilliance distribution of citation graph')
plt.plot(xdata, ydata, marker='.', linestyle='None', color='b')
plt.plot(xdata1, ydata1, marker='.', linestyle='None', color='y')
plt.savefig("na_brill0")
