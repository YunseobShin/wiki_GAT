from collections import Counter
import json
import numpy as np
import re, sys
from tqdm import tqdm

with_file = int(sys.argv[1])

sample_nodes = np.load('wiki_data/wiki_node_5k.npy')
f = open('wiki_data/wiki_dic.json').read()
wiki_dic = json.loads(f)
f = open('wiki_data/title_index.json').read()
t_i = json.loads(f)
f = open('wiki_data/index_title.json').read()
i_t = json.loads(f)

label_pattern = re.compile('\[\[Category:(.*?)\]\]')

if with_file != 1:
    t_l = {}
    print('catching labels...')
    for node in tqdm(sample_nodes):
        if node not in i_t:
            continue
        page = wiki_dic[i_t[node]]
        id = re.findall("<id>(.*?)</id>", page)
        if len(id) > 0:
            id = id[0]
        else:
            continue
        if id in sample_nodes:
            labels = label_pattern.findall(page)
            # print(labels)
            t_l[i_t[node]] = labels

    ls = []
    for v in tqdm(t_l.values()):
        for s in v:
            ls.append(s)

    counts = sorted(Counter(ls).items(), reverse=True, key=lambda kv: kv[1])
    np.save('labels', counts)

else:
    counts = np.load('labels.npy')
print(counts[:50])
si = int(input('start index: '))
ti = int(input('end index: '))
counts = counts[si:ti]
print(counts)

counts = np.array(counts)[:,0]
# print(counts)
print('labeling nodes...')

t_l = {}
for node in tqdm(sample_nodes):
    if node not in i_t:
        continue
    title = i_t[node]
    page = wiki_dic[i_t[node]]
    id = re.findall("<id>(.*?)</id>", page)
    if len(id) > 0:
        id = id[0]
    else:
        continue
    if id in sample_nodes:
        labels = label_pattern.findall(page)
        for label in labels:
            if label in counts:
                t_l[title] = label
                break

print('number of labeled data:', len(t_l))
fname = 'wiki_data/title_labels_5k.json'
with open(fname, 'w') as g:
    json.dump(t_l, g)















#
