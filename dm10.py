# -*- coding: utf-8 -*

import networkx as nx
from numpy import double
from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans  


#initialize a graph
G = nx.Graph()
G.clear()

#read data
# k = 20
# file_name = "web-NotreDame"
# f = open("web-NotreDame.txt", "r")

# k = 50
# file_name = "roadNet-CA"
# f = open("roadNet-CA.txt", "r")

k = 10
file_name = "soc-Epinions1"
f = open("soc-Epinions1.txt", "r")

# k = 2
# file_name = "ca-GrQc"
# f = open("../data/ca-GrQc.txt", "r")

# file_name = "Oregon-1"
# f = open("data/Oregon-1.txt", "r")

for line in f.readlines(): 
    if len(line) and (not line.startswith('#')):  
        i = int(line.split(" ")[0])
        o = int(line.split(" ")[1].split("\n")[0])
        # print(i + ":" +o)
        G.add_edge(i,o)
    else:
        continue
f.close()

print("L",len(list(G.nodes())))



def spectralClustering(G, k):
    # D = diag(sum(X, axis=0)) #degree matrix
    # L = D - X #laplacian matrix
    # sL = sparse.csr_matrix(L)
    sL = nx.laplacian_matrix(G).astype(double)
    print("done")
    # eigval, eigvec = linalg.eig(L）
    tolerance = 0.001
    eigval, eigvec = eigs(sL, k=k,tol=tolerance,which="SM")
    print("eigvec:", len(eigvec))
    idx = eigval.real.argsort() # Get indices of sorted eigenvalues
    eigvec = eigvec.real[:,idx] # Sort eigenvectors according to eigenvalues
    Y = eigvec[:,:k] # Keep the first k vectors
    # centroids, distortion = kmeans2(Y, k)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(Y)
    distortion = kmeans.labels_
    return distortion

# S = nx.adjacency_matrix(G).toarray().astype(double)

labels = spectralClustering(G, k)



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
    
    nodes = list(G.nodes())
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

S = countVertices(labels,G)
e = evaluation(S,G)
print("labels: ",e)


result_file_name = file_name + ".output"

# f = open(result_file_name,'r+')
# f.truncate()
# f.close()

with open(result_file_name, 'w') as fo:
    comment = "# "+file_name+" "+str(len(list(G.nodes())))+" "+str(len(list(G.edges())))+" "+str(k)+'\n'
    fo.writelines(comment)  
    for key in S:
        for val in S[key]:
            line = str(val)+" "+str(key)+"\n"
            fo.writelines(line)

# fo.close()