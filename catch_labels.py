from collections import Counter
import json
import numpy as np
import re, sys
from tqdm import tqdm

f = open('wiki_dic.json').read()
wiki_dic = json.loads(f)
f = open('title_index.json').read()
t_i = json.loads(f)
f = open('index_title.json').read()
i_t = json.loads(f)
sample_nodes = np.load('wiki_30k.npy')

label_pattern = re.compile('\[\[Category:(.*?)\]\]')
t_l = {}
print('catching labels...')
for node in tqdm(sample_nodes):
    node = str(node)
    if node not in i_t:
        continue
    page = wiki_dic[i_t[node]]
    id = re.findall("<id>(.*?)</id>", page)
    if len(id) > 0:
        id = id[0]
    else:
        continue
    if int(id) in sample_nodes:
        labels = label_pattern.findall(page)
        # print(labels)
        t_l[i_t[node]] = labels

ls = []
for v in tqdm(t_l.values()):
    for s in v:
        ls.append(s)

counts = sorted(Counter(ls).items(), reverse=True, key=lambda kv: kv[1])
np.save('labels', counts)
si = int(sys.argv[1])
ti = int(sys.argv[2])
print(len(counts))
print(counts[:50])
counts = counts[si:ti]
print(counts)
exit()
counts = np.array(counts)[:,0]
# print(counts)
print('labeling nodes...')

t_l = {}
for node in tqdm(sample_nodes):
    node = str(node)
    if node not in i_t:
        continue
    title = i_t[node]
    page = wiki_dic[i_t[node]]
    id = re.findall("<id>(.*?)</id>", page)
    if len(id) > 0:
        id = id[0]
    else:
        continue
    if int(id) in sample_nodes:
        labels = label_pattern.findall(page)
        for label in labels:
            if label in counts:
                t_l[title] = label
                break

print(len(t_l))
fname = 'title_labels.json'
with open(fname, 'w') as g:
    json.dump(t_l, g)















#
