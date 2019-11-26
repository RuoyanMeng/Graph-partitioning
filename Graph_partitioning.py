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
while True:
    line = f.readline()
    if not line.startswith('#'):   
        i = line.split("\t")[0]
        o = line.split("\t")[1].split("\n")[0]
        G.add_edge(i,o)
f.close()