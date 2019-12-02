import numpy as np
import networkx as nx
import numpy.linalg as la
import scipy.cluster.vq as vq
import matplotlib.pyplot as plt

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

# G = nx.powerlaw_cluster_graph(100, 1, 0.0)
coord = nx.spring_layout(G, iterations=10)

# fig = plt.figure()
# ax = fig.add_subplot(111, aspect='equal')
# ax.axis('off')
# nx.draw(G,pos=nx.spring_layout(G))
# plt.show()

# print("L: ",list(G.nodes)[:10])
# A = nx.adjacency_matrix(G)
# D = np.diag(np.ravel(np.sum(A,axis=1)))
# print("D: ",D[:10])

print(len(list(G.nodes)))



#Calculate graph Laplacian matrix
L = nx.laplacian_matrix(G)

# print("L: ",L.toarray()[:1000])

# a , u = la.eigh(L.toarray())
# print("a: ",a[:1000])


#compute covariance matri
C = np.cov(L.toarray())
# print("C:", len(C))
#compute eigenvalues/eigenvector
l, U = la.eigh(C)
print("U:", len(U))


# Fiedler vector
f = U[:,1]

# k-means
k = 2
means, labels = vq.kmeans2(U[:,1:k], k)


fig = plt.figure()
fig = nx.draw_networkx_nodes(G, coord,
node_size=2,
node_color=labels)
plt.show()

# #print(means)
# # print(np.unique(labels))
# # print(labels)

# #evaluation cluster
# # calculate |E(S,T)|
# def E_n(S,T,G):
#     # S: array, vertices in one of the cluster
#     # T: array, V - S, the vertices in graph except for vertices in S
#     # G: the graph
#     n = 0
#     for v in S:
#         # print(v)
#         for edge in list(G.edges(v)):
#             if (edge[1] in T):
#                 n += 1
#     print("lenE",n)
#     return n

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
#     S_inOne = []
    
#     for i in range(len(labels)):
#         node = nodes[i]
#         # S_inOne[i] = node
#         # S[i] = (labels[i],node)
#         if (labels[i] in S.keys()):
#             S[labels[i]].append((labels[i],node))
#         else:
#             S[labels[i]] = [node]
#     # print("S",S)
#     # np.sort(S, order='key')   
#     return S, S_inOne

# # evaluation function
# def evaluation(S,S_inOne,G,k):
#     # S: dict, cluster dict
#     # G: the graph
#     # k: number of cluster
#     frac = []
#     for key in S:
#         S_c = S[key]
#         T = list(set(S_inOne)-set(S_c))
#         edge_n = E_n(S_c,T,G)
#         frac.append(edge_n/len(S_c))
    
#     eval = sum(frac)
#     return eval

# S, S_inOne = countVertices(labels,G)
# e = evaluation(S,S_inOne,G,k)
# print(e)