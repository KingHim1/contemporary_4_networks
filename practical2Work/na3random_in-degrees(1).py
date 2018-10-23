#na3random_in-degrees.py
#matthew johnson 19 january 2017

#####################################################

import random

def make_random_graph(num_nodes, prob):
    """Returns a random undirected graph with the specified number of nodes and edge probability. 
    The nodes of the graph are numbered 0 to num_nodes - 1.  For every pair of nodes, i and j, the pair is connected with probability prob. """
    #initialize empty graph
    random_graph = {} 
    for vertex in range(num_nodes): random_graph[vertex] = []
    #consider each vertex
    for vertex in range(num_nodes):
        #consider each neighbour with greater value
        for neighbour in range(vertex + 1, num_nodes):
            random_number = random.random()
            #add edge from vertex to neighbour with probability prob
            if random_number < prob:
                #maybe this method for set union is deprecated in python 3
                random_graph[vertex] = set(random_graph[vertex]) | set([neighbour])        
                random_graph[neighbour] = set(random_graph[neighbour]) | set([vertex])        
    return random_graph


def degree_distribution(graph):
    """Takes a graph and computes the unnormalized distribution of the degrees of the nodes. 
    Returns a dictionary whose keys correspond to degrees of nodes in the graph and values are the number of nodes 
    with that degree. 
    Degrees with no corresponding nodes in the graph are not included in the dictionary."""
    #initialize dictionary for degree distribution
    degree_distribution = {}
    #consider each vertex
    for vertex in graph:
        #update degree_distribution
        if len(graph[vertex]) in degree_distribution:
            degree_distribution[len(graph[vertex])] += 1
        else:
            degree_distribution[len(graph[vertex])] = 1
    return degree_distribution
    
def normalized_degree_distribution(graph, num_nodes):
    """Takes a graph and computes the normalized distribution of the degrees of the graph. 
    Returns a dictionary whose keys correspond to in-degrees of nodes in the graph and values are the 
    fraction of nodes with that in-degree. 
    Degrees with no corresponding nodes in the graph are not included in the dictionary."""
    unnormalized_dist = degree_distribution(graph)
    normalized_dist = {}
    for degree in unnormalized_dist:
        normalized_dist[degree] = 1.0 * unnormalized_dist[degree] / num_nodes
    return normalized_dist

def average_normalized_degree_distribution(k):
    """finds the average degree distribution of a random graph on 2000 nodes
    with edge probability 0.1 by taking k trials for each data point"""
    cumulative_dist = {}
    for deg in range(140,260): cumulative_dist[deg] = 0
    for i in range(k):
        graph = make_random_graph(2000, 0.1)
        #find the distribution
        dist = normalized_degree_distribution(graph, 2000)
        for deg in range(140,260):
            if deg in dist:
                cumulative_dist[deg] += dist[deg]
    average_dist = {}
    for deg in range(140,260):
        average_dist[deg] = cumulative_dist[deg] / k
    return average_dist

#normalized degree distribution averaged over 100 trials (rather 25 asked for in the exercise)
averaged_degrees = average_normalized_degree_distribution(100)

    
#create arrays for plotting
xdata = []
ydata = []
for degree in averaged_degrees:
    xdata += [degree]
    ydata += [averaged_degrees[degree]]
    
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

#clears plot
plt.clf()

#plot degree distribution 
plt.xlabel('In-Degree')
plt.ylabel('Normalized Rate')
plt.title('Degree Distribution of Random Graph')
plt.plot(xdata, ydata, marker='.', linestyle='None', color='b')
plt.savefig('na3random.png')
