#na3random_giant_component.py
#matthew johnson 20 january 2017

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

def largest_component_size(graph):
    """returns the size of a graphs largest component by doing a depth first
    search"""
    found = {}
    for vertex in graph: found[vertex] = 0
    def search(vertex, comp_size):
        found[vertex] = 1
        for neighbour in graph[vertex]:
            if found[neighbour] == 0:
                comp_size = search(neighbour, comp_size)
        return comp_size + 1
    max_comp_size = 0
    for vertex in graph:
        if found[vertex] == 0:
            comp_size = search(vertex, 0)
            if comp_size > max_comp_size:
                max_comp_size = comp_size
    return max_comp_size

def largest_component_vs_edge_probability(k):
    """generates data of size of giant component of random graph to be plotted
    and takes average of k trials for each data point"""
    xdata = []
    ydata = []
    for i in range(1, 321):
        prob = i / 60000.0
        comp_sizes = []
        for j in range(k):
            example = make_random_graph(1000, prob)  
            comp_sizes += [largest_component_size(example)]
        xdata += [prob]
        ydata += [sum(comp_sizes)/k]
    return xdata, ydata

    
#create arrays for plotting
xdata, ydata = largest_component_vs_edge_probability(30) 
    
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

#clears plot
plt.clf()

#plot degree distribution 
plt.xlabel('edge probability')
plt.ylabel('size of component')
plt.title('The Giant Component of a Random Graph, n=1000')
plt.plot(xdata, ydata, marker='.', linestyle='None', color='b')
plt.savefig('na3randomcomponent.png')
