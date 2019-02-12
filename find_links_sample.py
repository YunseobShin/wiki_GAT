import numpy as np
import sys
from tqdm import tqdm
from time_check import tic
from time_check import toc
import multiprocessing
from multiprocessing import Pool

def write_sample(E):
    output = open('edges_30k.txt', 'a')
    sample_nodes = np.load('wiki_data/wiki_node_30k.npy')
    s=int(E[0])
    t=int(E[1])
    # print(s, t)
    if s in sample_nodes and t in sample_nodes:
        output.write(str(s)+' '+str(t)+'\n')
    output.close()


with open('wiki_data/edges.txt', 'r') as f:
    edges = f.readlines()

edges = [x.split(' ') for x in tqdm(edges)]
# print(edges[:20])
pool_size = 26
output = open('edges_30k.txt', 'w')
output.close()
pool = Pool(processes = pool_size)
tic()
pool.map(write_sample, edges)
toc()
