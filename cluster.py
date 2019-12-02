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

# A = nx.adjacency_matrix(G)
k = 2
A = nx.to_numpy_matrix(G)
n = np.shape(A)[0]
D = np.diag(1 / np.sqrt(np.ravel(A.sum(axis=0))))
L = np.identity(n) - D.dot(A).dot(D)


# L = nx.laplacian_matrix(G)
# L.asfptype()
print("LL:", type(L))
V, Z = eigsh(L, k, which='SM')

print("Z",len(Z))

rows_norm = np.linalg.norm(Z, axis=1, ord=2)
Y = (Z.T / rows_norm).T
centroids, distortion = kmeans(Y, k)
    
y_hat = np.zeros(n, dtype=int)
for i in range(n):
    dists = np.array([np.linalg.norm(Y[i] - centroids[c]) for c in range(k)])
    y_hat[i] = np.argmin(dists)

# print(y_hat[:1000])

fig = plt.figure()
nx.draw_networkx(G, with_labels=False,
                 node_color=y_hat, node_size=2)
plt.show()

#evaluation cluster
# calculate |E(S,T)|
def E_n(S,T,G):
    # S: array, vertices in one of the cluster
    # T: array, V - S, the vertices in graph except for vertices in S
    # G: the graph
    n = 0
    for v in S:
        # print(v)
        for edge in list(G.edges(v)):
            if (edge[1] in T):
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
    S_inOne = []
    
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
    return S, S_inOne

# evaluation function
def evaluation(S,S_inOne,G,k):
    # S: dict, cluster dict
    # G: the graph
    # k: number of cluster
    frac = []
    for key in S:
        S_c = S[key]
        T = list(set(S_inOne)-set(S_c))
        edge_n = E_n(S_c,T,G)
        frac.append(edge_n/len(S_c))
    
    eval = sum(frac)
    return eval

S, S_inOne = countVertices(y_hat,G)
e = evaluation(S,S_inOne,G,k)
print(e)