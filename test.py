import networkx as nx
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt


from scipy.cluster.vq import whiten, kmeans
from numpy import linalg as LA
from networkx.algorithms.cuts import conductance
from scipy.sparse.linalg import eigsh

#initialize a graph
G = nx.Graph()
G.clear()

#read data
f = open("data/ca-GrQc.txt", "r")

for line in f.readlines(): 
    if len(line) and (not line.startswith('#')):  
        i = int(line.split(" ")[0])
        o = int(line.split(" ")[1].split("\n")[0])
        # print(i + ":" +o)
        G.add_edge(i,o)
    else:
        continue
f.close()

print("L",len(list(G.nodes)))

def eig_laplacian(A, k=2):
    n = np.shape(A)[0]
    D = np.diag(1 / np.sqrt(np.ravel(A.sum(axis=0))))
    L = np.identity(n) - D.dot(A).dot(D)
    return eigsh(L, k, which='SM')

def spectral_clust(A, k=2):
    n = np.shape(A)[0]
    V, Z = eig_laplacian(A, k)
    
    rows_norm = np.linalg.norm(Z, axis=1, ord=2)
    Y = (Z.T / rows_norm).T
    centroids, distortion = kmeans(Y, k)
    
    y_hat = np.zeros(n, dtype=int)
    for i in range(n):
        dists = np.array([np.linalg.norm(Y[i] - centroids[c]) for c in range(k)])
        y_hat[i] = np.argmin(dists)
    return y_hat

# #
# graph = max(nx.connected_component_subgraphs(G), key=len)
    
# # Split the graph edges into train and test
# random_edges = list(graph.edges())
# np.random.shuffle(random_edges)
# train_edges = random_edges[:graph.number_of_edges()//2]
# test_edges = random_edges[graph.number_of_edges()//2:]
    
# # Create the training graph
# train_graph = nx.Graph()
# train_graph.add_edges_from(train_edges)
# train_graph = max(nx.connected_component_subgraphs(train_graph), key=len)
    
# # Create the test graph
# test_graph = nx.Graph()
# test_graph.add_nodes_from(train_graph.nodes())
# test_graph.add_edges_from(test_edges)

# N = train_graph.number_of_nodes()
# S = nx.to_numpy_matrix(train_graph)
# tao = train_graph.number_of_edges() * 2 / N
# SR = S + tao / N

S = nx.to_numpy_matrix(G)
SR = S + (G.number_of_edges() * 2 / np.shape(S)[0]**2)
        
van_labels = spectral_clust(S,10)
reg_labels = spectral_clust(SR,10)

#evaluation cluster
# calculate |E(S,T)|
def E_n(S,G):
    # S: array, vertices in one of the cluster
    # T: array, V - S, the vertices in graph except for vertices in S
    # G: the graph
    n = 0
    for v in S:
        # print(v)
        for edge in list(G.edges(v)):
            if (edge[1] not in S):
                n += 1
    print("lenE",n)
    return n

# get vertices in each cluster
def countVertices(labels,G):
    # labels: array, labels[i] is the cluster index i belongs to
    # k: int, cluster number
    
    nodes = list(G.nodes)
    n_nodes = len(nodes)
    # S = np.zeros((n_nodes,), dtype=('i4,i4'))
    # S.dtype.names = ('key', 'value')
    # S_inOne = np.zeros(n_nodes)
    S = dict()    
    for i in range(len(labels)):
        node = nodes[i]
        # S_inOne[i] = node
        # S[i] = (labels[i],node)
        if (labels[i] in S.keys()):
            S[labels[i]].append(node)
        else:
            S[labels[i]] = [node]
    # print("S",S)
    # np.sort(S, order='key')   
    return S

# evaluation function
def evaluation(S,G):
    # S: dict, cluster dict
    # G: the graph
    frac = []
    for key in S:
        S_c = S[key]
        # T = list(set(S_inOne)-set(S_c))
        edge_n = E_n(S_c,G)
        frac.append(edge_n/len(S_c))
    print(frac)
    eval = sum(frac)
    return eval

S = countVertices(reg_labels,G)
e = evaluation(S,G)
print("reg_labels: ",e)

van_labels
S = countVertices(van_labels,G)
e_v = evaluation(S,G)

print("van_labels: ", e_v)


# fig = plt.figure()
# nx.draw_networkx(G, with_labels=False,
#                  node_color=reg_labels, node_size=2)
# plt.show()

