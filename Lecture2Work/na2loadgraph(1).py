#na2loadgraph.py
#matthew johnson 19 january 2017

#####################################################

"""
The following code reads the file and creates the citation network as a digraph.
"""

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


