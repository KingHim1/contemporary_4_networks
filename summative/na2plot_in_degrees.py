#na2plot_in-degrees.py
#matthew johnson 19 january 2017

#####################################################

def compute_in_degrees(digraph):
    """Takes a directed graph and computes the in-degrees for the nodes in the
    graph. Returns a dictionary with the same set of keys (nodes) and the
    values are the in-degrees."""
    #initialize in-degrees dictionary with zero values for all vertices
    in_degree = {}
    for vertex in digraph:
        in_degree[vertex] = 0
    #consider each vertex
    for vertex in digraph:
        #amend in_degree[w] for each outgoing edge from v to w
        for neighbour in digraph[vertex]:
            in_degree[neighbour] += 1
    return in_degree

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


def load_graph(graph_txt):
    """
    Loads a graph from a text file.
    Then returns the graph as a dictionary.
    """
    graph = open(graph_txt)
    
    answer_graph = {}
    nodes = 0
    for line in graph:
        neighbors = line.split(' ')
        node = int(neighbors[0])
        answer_graph[node] = set([])
        for neighbor in neighbors[1 : -1]:
            answer_graph[node].add(int(neighbor))
        nodes += 1
    print ("Loaded graph with", nodes, "nodes")

    return answer_graph

citation_graph = load_graph("alg_phys-cite.txt")

citations_in_degrees = compute_in_degrees(citation_graph)

citations_distribution = in_degree_distribution(citation_graph)

#How many papers are cited just once?  
print (citations_distribution[1])

#Which paper is cited the most and how many times is it cited?
most_cited = max(citations_in_degrees, key=lambda k: citations_in_degrees[k])
print(most_cited, citations_in_degrees[most_cited])


normalized_citations_distribution = {}
for degree in citations_distribution:
    normalized_citations_distribution[degree] = citations_distribution[degree] / 27770.0

#create arrays for plotting
xdata = []
ydata = []
for degree in normalized_citations_distribution:
    xdata += [degree]
    ydata += [normalized_citations_distribution[degree]]
    
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

#plot degree distribution

plt.xlabel('In-Degree')
plt.ylabel('Normalized Rate')
plt.title('In-Degree Distribution of Citation Graph')
plt.loglog(xdata, ydata, marker='.', linestyle='None', color='b')
plt.savefig('na2.png')
    



