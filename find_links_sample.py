import numpy as np
import networkx as nx
from tqdm import tqdm

g = nx.read_edgelist('wiki_data/edges.txt', create_using=nx.Graph())
sample_nodes = np.load('wiki_data/wiki_node_5k.npy')
sample_graph = g.subgraph(sample_nodes)

edges = list(sample_graph.edges())
output = open('wiki_data/edges_5k.txt', 'w')
for e in tqdm(edges):
    e = str(e)
    e = e.replace('(', '')
    e = e.replace(')', '')
    e = e.replace("'", "")
    e = e.replace(',', '')
    output.write(e+'\n')

output.close()
