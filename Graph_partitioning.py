import numpy as np
import networkx as nx
import numpy.linalg as la
import scipy.cluster.vq as vq
import matplotlib.pyplot as plt

#initialize a graph
G = nx.Graph()
G.clear()

#read data
f = open("data/web-NotreDame.txt", "r")
line = f.readline()
n = 0
for line in f.readlines(): 
    line.rstrip("\n") 

    if len(line) and (not line.startswith('#')):  
        i = line.split("\t")[0]
        o = line.split("\t")[1]
        # print(i + ":" +o)
        G.add_edge(i,o)
    else:
        continue
f.close()

#Calculate graph Laplacian matrix
L = nx.laplacian_matrix(G)
