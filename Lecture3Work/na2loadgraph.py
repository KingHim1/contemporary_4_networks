#na2loadgraph.py
#matthew johnson 19 january 2017

#####################################################

"""
The following code reads the file and creates the citation network as a digraph.
"""

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


in_degrees = compute_in_degrees(citation_graph)
count = 0
for x in in_degrees:
    if in_degrees[x] == 1:
        count += 1
print(count)

