#na5ws.py
#matthew johnson
#27 january 2015, last edited 2 february 2017

import random
import queue

def make_ws_graph(num_nodes, clockwise_neighbours, rewiring_prob):
    """Returns a dictionary to a undirected graph with num_nodes nodes. 
    The nodes of the graph are numbered 0 to num_nodes - 1.
    Node i initially joined to i+1, i+2, ... , i+d mod N 
    Each edge replaced with probability given with edge from i to randomly chosen k
    """
    #initialize empty graph
    ws_graph = {}
    for vertex in range(num_nodes): ws_graph[vertex] = []
    #add each vertex to clockwise neighbours
    for vertex in range(num_nodes):
        for neighbour in range(vertex + 1, vertex + clockwise_neighbours + 1):
            neighbour = neighbour % num_nodes
            ws_graph[vertex] += [neighbour]
            ws_graph[neighbour] += [vertex]
    for vertex in range(num_nodes):
        for neighbour in ws_graph[vertex]:
            if random.random() < rewiring_prob:
                ws_graph[vertex].remove(neighbour)
                ws_graph[neighbour].remove(vertex)
                randNode = random.randint(0, num_nodes-1)
                while(vertex == randNode):
                    randNode = random.randint(0, num_nodes - 1)
                ws_graph[vertex] += [randNode]
                ws_graph[randNode] += [vertex]


    return ws_graph
    #rewire each edge with probability rewiring_prob

    #consider each vertex

    #consider each neighbour

    #decide whether to rewire and join to a random node

    #update if necessary

"""to decide whether to rewire use

            random_number = random.random()
            if random_number < rewiring_prob:

to choose a random node use

            random_node = random.randint(0, num_nodes-1)
"""



def local_clustering_coefficient(graph, vertex):
    """returns ratio of edges to possible edges in neighbourhood of vertex"""
    edge_count = 0
    for neighbour1 in graph[vertex]:
        for neighbour2 in graph[vertex]:                                        #look at each pair of neighbours of vertex
            if neighbour1 in graph[neighbour2]:                                 #if the neighbours are joined to each other by an edge
                edge_count += 1                                                 #add one to the edge count
    degree = len(graph[vertex])                                                 #count how many neighbours vertex has
    return edge_count / (degree * (degree - 1))                                 #note factor of 2 missing as each edge counted twice



def clustering_coefficient(graph):
    """returns average of local clustering coefficients"""
    count = 0
    sumOfClusteringCoefficients = 0
    for vertex in graph:
        count += 1
        sumOfClusteringCoefficients += local_clustering_coefficient(graph, vertex)
    return sumOfClusteringCoefficients / count



def max_dist(graph, source):
    """finds the distance (the length of the shortest path) from the source to
    every other vertex in the same component using breadth-first search, and
    returns the value of the largest distance found"""
    q = queue.Queue()
    found = {}
    distance = {}
    for vertex in graph: 
        found[vertex] = 0
        distance[vertex] = -1
    max_distance = 0
    found[source] = 1
    distance[source] = 0
    q.put(source)
    while q.empty() == False:
        current = q.get()
        for neighbour in graph[current]:
            if found[neighbour] == 0:
                found[neighbour] = 1
                distance[neighbour] = distance[current] + 1
                max_distance = distance[neighbour]
                q.put(neighbour)
    return max_distance
    
def diameter(graph):
    """returns the diameter of a graph by finding greatest max distance"""
    max_distance = 0
    for vertex in graph:
        new_dist = max_dist(graph, vertex)
        if new_dist > max_distance:
            max_distance = new_dist
    return max_distance

def diameter_clustering_vs_prob_ws(num_nodes, k):
    """diameter and clustering coefficient vs rewiring prob with k trials"""
    xdata = []
    ydata = []
    zdata = []
    prob = 0.0005
    while prob < 1:
        xdata += [prob]
        diameters = []
        coeffs = []
        for i in range(k):
            graph = make_ws_graph(num_nodes, 8, prob)
            diameters += [diameter(graph)]
            coeffs += [clustering_coefficient(graph)]
        ydata += [sum(diameters) / k / 19.0] #divide by 19 as this diameter of circle lattice
        zdata += [sum(coeffs) / k / 0.7] #divide by 0.7 as this is clustering coefficient of circle lattice
        prob = 1.2*prob
    return xdata, ydata, zdata

#look at past exercises for plotting

# print(diameter_clustering_vs_prob_ws(100, 5))

results = diameter_clustering_vs_prob_ws(300, 8)

print(results[2])

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt



plt.xlabel('Rewiring Probability')
plt.ylabel('Diameters')
plt.title('Diameter Distribution of ws Graph')
plt.semilogx(results[0], results[1], marker='.', linestyle='-', color='b')
plt.semilogx(results[0], results[2], marker='.', linestyle='-', color='r')
plt.savefig('na4_ws_diam.png')

