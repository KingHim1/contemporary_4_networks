#na3pa.py
#matthew johnson 19 january 2017

#####################################################

"""the following code creates a PA graph and is based on code from
http://www.codeskulptor.org/#alg_dpa_trial.py; """

import random
from na2loadgraph import *

#first we need
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
        self._num_nodes = num_nodes #note that the vertices are labelled from 0 so self._num_nodes is the label of the next vertex to be added
        self._node_numbers = [node for node in range(num_nodes) for dummy_idx in range(num_nodes)]
        print(self._node_numbers)

    def run_trial(self, num_nodes):
        """
        Conduct num_node trials using by applying random.choice()
        to the list of node numbers
        
        Updates the list of node numbers so that the number of instances of
        each node number is in the same ratio as the desired probabilities
        
        Returns:
        Set of nodes
        """
        print(self._node_numbers)
        #compute the neighbors for the newly-created node
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
        #print(self._node_numbers)
        return new_node_neighbors
    
def make_complete_graph(num_nodes):
    """Takes the number of nodes num_nodes and returns a dictionary
    corresponding to a complete directed graph with the specified number of
    nodes. A complete graph contains all possible edges subject to the
    restriction that self-loops are not allowed. The nodes of the graph should
    be numbered 0 to num_nodes - 1 when num_nodes is positive. Otherwise, the
    function returns a dictionary corresponding to the empty graph."""
    #initialize empty graph
    complete_graph = {}
    #consider each vertex
    for vertex in range(num_nodes):
        #add vertex with list of neighbours
        complete_graph[vertex] = set([j for j in range(num_nodes) if j != vertex])
    return complete_graph
    
def make_PA_Graph(total_nodes, out_degree):
    """creates a PA_Graph on total_nodes where each vertex is iteratively
    connected to a number of existing nodes equal to out_degree"""
    #initialize graph by creating complete graph and trial object
    PA_graph = make_complete_graph(out_degree)
    trial = PATrial(out_degree)
    for vertex in range(out_degree, total_nodes):
        PA_graph[vertex] = trial.run_trial(out_degree)
    return PA_graph



## King - added Code below -- draw indegree distribution for PA_Graph
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


rand_graph = make_PA_Graph(100, 4)

rand_in_degrees = compute_in_degrees(rand_graph)

rand_distribution = in_degree_distribution(rand_graph)

normalized_rand_citations_distribution = {}
for degree in rand_distribution:
    normalized_rand_citations_distribution[degree] = rand_distribution[degree] / 2000




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
plt.loglog(xdata, ydata, marker='.', linestyle='None', color='b')
plt.savefig('na3_rand.png')