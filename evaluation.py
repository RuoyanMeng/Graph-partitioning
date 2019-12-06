
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
file_name = "data/Oregon-1"
f = open("data/Oregon-1.txt", "r")

for line in f.readlines(): 
    if len(line) and (not line.startswith('#')):  
        i = int(line.split(" ")[0])
        o = int(line.split(" ")[1].split("\n")[0])
        # print(i + ":" +o)
        G.add_edge(i,o)
    else:
        continue
f.close()

nodes_nn = len(list(G.nodes))
print("L",nodes_nn)


#read data

f = open("data/Oregon-1.output", "r")

S = dict()
for line in f.readlines(): 
    if len(line) and (not line.startswith('#')):  
        i = int(line.split(" ")[0])
        o = int(line.split(" ")[1].split("\n")[0])
        if o in S.keys():
            S[o].append(i)
        else:
            S[o] = [i]
    else:
        continue
f.close()




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

# # get vertices in each cluster
# def countVertices(labels,G):
#     # labels: array, labels[i] is the cluster index i belongs to
#     # k: int, cluster number
    
#     nodes = list(G.nodes)
#     n_nodes = len(nodes)
#     # S = np.zeros((n_nodes,), dtype=('i4,i4'))
#     # S.dtype.names = ('key', 'value')
#     # S_inOne = np.zeros(n_nodes)
#     S = dict()    
#     for i in range(len(labels)):
#         node = nodes[i]
#         # S_inOne[i] = node
#         # S[i] = (labels[i],node)
#         if (labels[i] in S.keys()):
#             S[labels[i]].append(node)
#         else:
#             S[labels[i]] = [node]
#     # print("S",S)
#     # np.sort(S, order='key')   
#     return S

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

# S = countVertices(e,G)
print(S[0])
print(S[1])

e = evaluation(S,G)
print("e: ",e)