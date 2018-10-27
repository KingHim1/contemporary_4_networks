import random
import queue
import time

def make_ring_graph(m,k,p,q):
    x = time.clock()
    ring_graph = {}
    for vertex in range(m * k):
        ring_graph[vertex] = []
    #Iterate through groups while not repeating group pairs
    for groupInd1 in range(0,m):
        for groupInd2 in range(groupInd1,m):
            isNeighbour = False
            isSame = False
    #Iterate through the vertices in each group, not repeating vertex pairs
            if groupInd1 == groupInd2:
                isSame = True
                isNeighbour = True
            if abs(groupInd1 - groupInd2) == 1:
                isNeighbour = True
            for ind1 in range(0, k):
                start = 0
                if isSame:
                    start = ind1
                for ind2 in range(start, k):
                    i = groupInd1 * k + ind1
                    j = groupInd2 * k + ind2
                    prob = random.random()
                    if isNeighbour and i != j:
                        if prob < p:
                            ring_graph[i] += [j]
                            ring_graph[j] += [i]
                    elif i != j:
                        if prob < q:
                            ring_graph[i] += [j]
                            ring_graph[j] += [i]
    print(time.clock() - x)
    return ring_graph

def make_ring_graph1(m, k, p, q):
    x = time.clock()
    ring_graph = {}
    for vertex in range(m * k):
        ring_graph[vertex] = []
    for i in range(m * k):
        for j in range(i, m * k):
            prob = random.random()
            if i / m == j / m or abs(i / m - j / m) == 1:
                if prob < p:
                    ring_graph[i] += [j]
                    ring_graph[j] += [i]
            else:
                if prob < q:
                    ring_graph[i] += [j]
                    ring_graph[j] += [i]
    print(time.clock() - x)
    return ring_graph


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


def local_clustering_coefficient(graph, vertex):
    """returns ratio of edges to possible edges in neighbourhood of vertex"""
    #look at each pair of neighbours of vertex
    ki = len(graph[vertex])
    ei = 0
    for neighbour in graph[vertex]:
        ei += len(graph[neighbour])
    return (2 * ei) / (ki * (ki - 1))

    #look at whether neighbour pair are joined by an edge

    #divide number of edges found by number of pairs considered



def clustering_coefficient(graph):
    """returns average of local clustering coefficients"""
    count = 0
    sumOfClusteringCoefficients = 0
    for vertex in graph:
        count += 1
        sumOfClusteringCoefficients += local_clustering_coefficient(graph, vertex)
    return sumOfClusteringCoefficients / count

def compute_in_degrees(graph):
    in_degree = {}
    for vertex in graph:
        in_degree[vertex] = len(graph[vertex])
    return in_degree

def in_degree_distribution(graph):
    #find in_degrees
    in_degree = compute_in_degrees(graph)
    #initialize dictionary for degree distribution
    degree_distribution = {}
    #consider each vertex
    for vertex in in_degree:
        #update degree_distribution
        if in_degree[vertex] in degree_distribution:
            degree_distribution[in_degree[vertex]] += 1
        else:
            degree_distribution[in_degree[vertex]] = 1
    return degree_distribution

def normalize_in_deg_dist(in_deg_dist, graph):
    normalized_citations_distribution = {}
    for degree in in_deg_dist:
        normalized_citations_distribution[degree] = in_deg_dist[degree] / len(graph.keys())
    return normalized_citations_distribution


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
        ydata += [sum(diameters) / k / 19.0]  # divide by 19 as this diameter of circle lattice
        zdata += [sum(coeffs) / k / 0.7]  # divide by 0.7 as this is clustering coefficient of circle lattice
        prob = 1.1 * prob
    return xdata, ydata, zdata


def diameter_clustering_vs_prob_ring(m, k, p, q, trials):
    """diameter and clustering coefficient vs rewiring prob with k trials"""
    xdata = []
    ydata = []
    zdata = []
    prob = 0.1
    for x in range(0, 1):
        xdata += [prob]
        diameters = []
        coeffs = []
        for i in range(trials):
            graph = make_ring_graph(m, k, p, q)
            diameters += [diameter(graph)]
            coeffs += [clustering_coefficient(graph)]
        ydata += [sum(diameters) / trials / 19.0]  # divide by 19 as this diameter of circle lattice
        zdata += [sum(coeffs) / trials / 0.7]  # divide by 0.7 as this is clustering coefficient of circle lattice
        prob = 1.1 * prob
        print(prob)
    return xdata, ydata, zdata


ring_graph = make_ring_graph(100, 10, 0.4, 0.1)

in_degrees = compute_in_degrees(ring_graph)

in_degree_dist = in_degree_distribution(ring_graph)

normalised_in_deg_dist = normalize_in_deg_dist(in_degree_dist, ring_graph)



xdata = []
ydata = []
for degree in normalised_in_deg_dist:
    xdata += [degree]
    ydata += [normalised_in_deg_dist[degree]]

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# plot degree distribution

plt.xlabel('In-Degree')
plt.ylabel('Normalised Rate')
plt.title('In-Degree Distribution of Ring Graph')
plt.loglog(xdata, ydata, marker='.', linestyle='None', color='b')
plt.savefig('na_ring_in_deg.png')

#
# results = diameter_clustering_vs_prob_ring(100, 10, 0.5, 0.1, 1)
#
# plt.xlabel('Rewiring Probability')
# plt.ylabel('Diameters')
# plt.title('Diameter Distribution of ws Graph')
# plt.plot(results[0], results[1], marker='.', linestyle='None', color='b')
# plt.savefig('na_ring_cluster.png')
