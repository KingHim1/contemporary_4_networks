import networkx as nx
import random
import queue
from summativeQuestion1 import make_ring_graph


class PATrial:
    """
    Used when each new node is added in creation of a PA graph.
    Maintains a list of node numbers with multiple instances of each number.
    The number of instances of each node number are in proportion to the
    probability that it is linked to.
    Uses random.choice() to select a node number from this list for each trial.
    """

    def __init__(self, num_nodes):
        """
        Initialize a PATrial object corresponding to a
        complete graph with num_nodes nodes

        Note the initial list of node numbers has num_nodes copies of
        each node number
        """
        self._num_nodes = num_nodes  # note that the vertices are labelled from 0 so self._num_nodes is the label of the next vertex to be added
        self._node_numbers = [node for node in range(num_nodes) for dummy_idx in range(num_nodes)]

    def run_trial(self, num_nodes):
        """
        Conduct num_node trials using by applying random.choice()
        to the list of node numbers

        Updates the list of node numbers so that the number of instances of
        each node number is in the same ratio as the desired probabilities

        Returns:
        Set of nodes
        """
        # compute the neighbors for the newly-created node
        # print(self._node_numbers)
        new_node_neighbors = set()
        for dummy_idx in range(num_nodes):
            new_node_neighbors.add(random.choice(self._node_numbers))
        # update the list of node numbers so that each node number
        # appears in the correct ratio
        self._node_numbers.extend(list(new_node_neighbors))
        # also add to the list of node numbers the id of the current node
        # since each node must appear once in the list else no future node will link to it
        # note that self._node_numbers will next be incremented
        self._node_numbers.append(self._num_nodes)
        # update the number of nodes
        self._num_nodes += 1
        # print(self._node_numbers)
        return new_node_neighbors

#
def make_complete_graph(num_nodes):
    """Takes the number of nodes num_nodes and returns a dictionary
    corresponding to a complete directed graph with the specified number of
    nodes. A complete graph contains all possible edges subject to the
    restriction that self-loops are not allowed. The nodes of the graph should
    be numbered 0 to num_nodes - 1 when num_nodes is positive. Otherwise, the
    function returns a dictionary corresponding to the empty graph."""
    # initialize empty graph
    complete_graph = {}
    # consider each vertex
    for vertex in range(num_nodes):
        # add vertex with list of neighbours
        complete_graph[vertex] = set([j for j in range(num_nodes) if j != vertex])
    return complete_graph


def make_PA_Graph(total_nodes, out_degree):
    """creates a PA_Graph on total_nodes where each vertex is iteratively
    connected to a number of existing nodes equal to out_degree"""
    # initialize graph by creating complete graph and trial object
    PA_graph = make_complete_graph(out_degree)
    trial = PATrial(out_degree)
    for vertex in range(out_degree, total_nodes):
        PA_graph[vertex] = trial.run_trial(out_degree)
    return PA_graph


# file = open("coauthorship.txt")
# text = file.read()
# text = text.split("*")
#
# verticesFromText = text[1].split("\n")
# verticesToName = {}
# cite_graph = nx.Graph()
# for i in range(1, len(verticesFromText)-1):
#     verticesToName[i] = verticesFromText[i].split("\"")[1]
#     cite_graph.add_node(i)
#
# edgesFromText = text[2].split("\n")
# for i in range(1, len(edgesFromText)-1):
#     edges = edgesFromText[i].split(" ")
#     # print(edges[2] + ": " + edges[3])
#     if int(edges[2]) != int(edges[3]):
#         cite_graph.add_edge(int(edges[2]), int(edges[3]))
# #
# xdata = []
# ydata = []
# cite_dist = {}
# for vertex in cite_graph.adj:
#     # xdata.append(vertex)
#     if len(cite_graph.adj[vertex]) != 0:
#         H = cite_graph.subgraph(cite_graph.adj[vertex]).to_undirected()
#         # print(nx.maximal_independent_set(H))
#         val = len(nx.maximal_independent_set(H))
#         # ydata.append(val)
#         if val in cite_dist:
#             cite_dist[val] += 1
#         else:
#             cite_dist[val] = 1
#     # else:
#         # ydata.append(0)
#

ring_graph = nx.Graph()
H = make_ring_graph(30, 100, 0.8, 0.2)
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


#
# commented here ->


# for y in cite_dist:
#     xdata.append(y)
#     ydata.append(cite_dist[y])
# #

# PA_graph = nx.Graph()
# PA_digraph = make_PA_Graph(1500, 10)
# for x in PA_digraph:
#     for y in PA_digraph[x]:
#         PA_graph.add_edge(int(x), int(y))
# # print(PA_digraph)
# print(PA_graph.adj)
# print(cite_graph.adj)
#
# xdata2 = []
# ydata2 = []
# PA_dist = {}
# for vertex in PA_graph.adj:
#     # xdata1.append(vertex)
#     if len(PA_graph.adj[vertex]) != 0:
#         H = PA_graph.subgraph(PA_graph.adj[vertex]).to_undirected()
#         # print(nx.maximal_independent_set(H))
#         val = len(nx.maximal_independent_set(H))
#         # print(val)
#         # ydata1.append(val)
#         if val in PA_dist:
#             PA_dist[val] +=1
#         else:
#             PA_dist[val] = 1
#     # else:
#     #     ydata1.append(0)
#
# for x in PA_dist:
#     xdata2.append(x)
#     ydata2.append(PA_dist[x])

import matplotlib.pyplot as plt
plt.xlabel('node index')
plt.ylabel('Brilliance')
plt.title('Brilliance distribution of citation graph')
# plt.loglog(xdata, ydata, marker='.', linestyle='None', color='b')
plt.loglog(xdata1, ydata1, marker='.', linestyle='None', color='y')
# plt.loglog(xdata2, ydata2, marker='.', linestyle='None', color='r')
plt.savefig("na_brill0")