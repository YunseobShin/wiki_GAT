import networkx as nx
import numpy as np
from time_check import tic
from time_check import toc

def take_second(e):
    return e[1]

graph_path='wiki_data/edges.txt'

g = nx.read_edgelist(graph_path, create_using=nx.Graph())
print('number of nodes:', g.number_of_nodes())

sample_size = int(input('sample size: '))
output_file='wiki_data/wiki_node_'+str(int(sample_size/1000))+'k'
print('Sampling...')
tic()
sample = np.array(sorted(list(g.degree()),
         reverse = True, key = take_second))[:, 0][:sample_size]
toc()
np.save(output_file, sample)
