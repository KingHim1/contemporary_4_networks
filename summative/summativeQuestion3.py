import random
import queue
import time
import networkx as nx
import math

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
        # print(self._node_numbers)

    def run_trial(self, num_nodes):
        """
        Conduct num_node trials using by applying random.choice()
        to the list of node numbers

        Updates the list of node numbers so that the number of instances of
        each node number is in the same ratio as the desired probabilities

        Returns:
        Set of nodes
        """
        # print(self._node_numbers)
        # compute the neighbors for the newly-created node
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


def make_ring_graph(m,k,p,q):
    # x = time.clock()
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
    # print(time.clock() - x)
    return ring_graph

def to_netx_graph(graph):
    nx_graph = nx.Graph()
    for x in graph:
        if len(graph[x]) == 0:
            nx_graph.add_node(x)
        for y in graph[x]:
            if y != x:
                nx_graph.add_edge(int(x), int(y))
    return nx_graph

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



def randomise_neighbours(neighbours):
    random.shuffle(neighbours)
    return(neighbours)

def rand_search(s, t, rand_graph):
    # print("NEW SEARCH")
    curr_node = s
    query_count = 0
    count = 0
    while curr_node != t:
        # print("neighbours of t: " + ' '.join(str(e) for e in list(rand_graph.neighbors(t))))
        # print(list(rand_graph.neighbors(curr_node)))
        count += 1
        neighbours = randomise_neighbours(list(rand_graph.neighbors(curr_node)))
        # print(neighbours)
        # print(rand_graph.adj)
        for neighbour in neighbours:
            query_count += 1
            if neighbour == t:
                print(count)
                return query_count, count
        if query_count >= 1000 or len(neighbours) == 0:
            return query_count, count
        curr_node = random.choice(neighbours)
    return query_count, count

# trials = 1000
# div = trials
# sum = 0
# sumc = 0
# for x in range(0, trials):
#     random_graph = to_netx_graph(make_random_graph(50, 0.1))
#     count = rand_search(0, 10, random_graph)
#     print(count)
#     c = count[1]
#     count = count[0]
#     if count >= 1000:
#         div -= 1
#     else:
#         sum += count
#         sumc += c
# print(sumc / div)
# print(sum / div)

def ring_query_node(node, k):
    return node, node / k

def ring_search(s, t, ring_graph, group_size):
    k = group_size
    curr_node = s
    query_count = 0
    tar_node_group = t / k
    while curr_node != t:
        neighbours = randomise_neighbours(list(ring_graph.neighbors(curr_node)))
        potential_next_node = []
        dist_of_closest = abs(neighbours[0]/group_size - tar_node_group)
        closest_to_tar_group = neighbours[0]
        for neighbour in neighbours:
            query_count += 1
            dist_of_neigh_group = abs(neighbour/group_size - tar_node_group)
            if neighbour == t:
                return query_count
            if dist_of_neigh_group == 1:
                potential_next_node.append(neighbour)
            elif dist_of_neigh_group < dist_of_closest:
                dist_of_closest = dist_of_neigh_group
                closest_to_tar_group = neighbour
        if len(potential_next_node) != 0:
            curr_node = random.choice(potential_next_node)
        else:
            #Move to a node that's group is closest to the target groups
            curr_node = closest_to_tar_group
    return query_count

def ring_search_loc_est(s, t, ring_graph, group_size):
    k = group_size
    curr_node = s
    query_count = 0
    tar_node_group = t / k
    while curr_node != t:
        neighbours = randomise_neighbours(list(ring_graph.neighbors(curr_node)))
        potential_next_nodes = []
        dist_of_closest = abs(neighbours[0] / group_size - tar_node_group)
        closest_to_tar_group = neighbours[0]
        for neighbour in neighbours:
            query_count += 1
            dist_of_neigh_group = abs(neighbour / group_size - tar_node_group)
            if neighbour == t:
                return query_count
            if dist_of_neigh_group == 1:
                potential_next_nodes.append(neighbour)
            elif dist_of_neigh_group < dist_of_closest:
                dist_of_closest = dist_of_neigh_group
                closest_to_tar_group = neighbour
        num_of_non_adj_nodes = len(neighbours) - len(potential_next_nodes)
        num_nodes_estimate = max(neighbours)

        #get p estimate and q estimate - not yet complete
        q_estimate = num_of_non_adj_nodes/num_nodes_estimate
        p_estimate = len(potential_next_nodes)/num_nodes_estimate

        #if p < q go to q
        if len(potential_next_nodes) != 0:
            curr_node = random.choice(potential_next_nodes)
        else:
            # Move to a node that's group is closest to the target groups
            curr_node = closest_to_tar_group
    return query_count

def PA_search(s, t, PA_graph, m):
    curr_node = s
    query_count = 0
    while curr_node != t:
        #Query up to any neighbour with id <= m else move to lowest id
        neighbours = randomise_neighbours(list(PA_graph.neighbors(curr_node)))
        potential_next_node = []
        for neighbour in neighbours:
            query_count += 1
            if neighbour == t:
                return query_count
            if neighbour <= m:
                potential_next_node.append(neighbour)
        if len(potential_next_node)!=0:
            curr_node = random.choice(potential_next_node)
        else:
            curr_node = random.choice(neighbours)
    return query_count

# PA_graph = to_netx_graph(make_PA_Graph(200, 10))
#
# print(PA_search(100, 102, PA_graph, 10))

ring_graph = to_netx_graph(make_ring_graph(100, 10, 0.4, 0.1))
print(ring_search(10, 40, ring_graph, 10))