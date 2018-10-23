#na2random.py
#matthew johnson 19 january 2017

#####################################################

import random
from na2loadgraph import *

def make_random_graph(num_nodes, prob):
    """Returns a dictionary to a random graph with the specified number of nodes
    and edge probability.  The nodes of the graph are numbered 0 to
    num_nodes - 1.  For every pair of nodes, i and j, the pair is considered
    twice: once to add an edge (i,j) with probability prob, and then to add an
    edge (j,i) with probability prob. 
    """
    #initialize empty graph
    random_graph = {}
    #consider each vertex
    for vertex in range(num_nodes):
        out_neighbours = []
        for neighbour in range(num_nodes):
            if vertex != neighbour:
                random_number = random.random()
                if random_number < prob:
                    out_neighbours += [neighbour]        
        #add vertex with list of out_ neighbours
        random_graph[vertex] = set(out_neighbours)
    return random_graph

# King - added code below - print normalised in degree distribution graph for random graph


def in_degree_distribution(digraph):
    """Takes a directed graph and computes the unnormalized distribution of the
    in-degrees of the graph.  Returns a dictionary whose keys correspond to
    in-degrees of nodes in the graph and values are the number of nodes with
    that in-degree. In-degrees with no corresponding nodes in the graph are not
    included in the dictionary."""
    #find in_degrees
    in_degree = compute_in_degrees(digraph)
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


listOfDists = []
numOfDists = 25
for i in range(0, numOfDists):
    rand_graph = make_random_graph(2000, 0.1)

    rand_in_degrees = compute_in_degrees(rand_graph)

    rand_distribution = in_degree_distribution(rand_graph)

    listOfDists.append(rand_distribution)


normalized_rand_citations_distribution = {}
for distr in listOfDists:
    for degree in distr:
        if normalized_rand_citations_distribution.get(degree) == None:
            normalized_rand_citations_distribution[degree] = distr[degree]
        else:
            normalized_rand_citations_distribution[degree] += distr[degree]

for degree in normalized_rand_citations_distribution:
    normalized_rand_citations_distribution[degree] = normalized_rand_citations_distribution[degree] / (numOfDists * 2000)




# create arrays for plotting
xdata = []
ydata = []
for degree in normalized_rand_citations_distribution:
    xdata += [degree]
    ydata += [normalized_rand_citations_distribution[degree]]

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# plot degree distribution

plt.xlabel('In-Degree')
plt.ylabel('Normalized Rate')
plt.title('In-Degree Distribution of Citation Graph')
plt.plot(xdata, ydata, marker='.', linestyle='None', color='b')
plt.savefig('na2_rand.png')