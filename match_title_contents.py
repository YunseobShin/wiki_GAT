import json, re
from tqdm import tqdm
import numpy as np

sample_nodes = np.load('wiki_data/wiki_node_5k.npy')
sample_nodes = [int(x) for x in sample_nodes]
fn = open('wiki_data/wiki', 'r')
wiki = fn.read()
docs = wiki.split('</doc>')
title_contents={}

for doc in tqdm(docs):
    if len(re.findall('id=\"(.*?)\"', doc)) > 0:
        id = re.findall('id=\"(.*?)\"', doc)[0]
    else:
        continue

    if int(id) in sample_nodes:
        if len(re.findall('title=\"(.*?)\"', doc)) > 0:
            title = re.findall('title=\"(.*?)\"', doc)[0]
            title = title.replace(' ', '_')
            title = title.replace('/', '-')
        else:
            continue
        # print(title)
        contents = re.sub('<doc(.*?)>', '', doc)
        contents = contents.replace('\n', ' ')
        contents = re.sub(r'[\W_]+', ' ', contents)
        contents = " ".join(contents.split())
        contents = contents.lower()
        contents = contents.replace(' a ', ' ')
        title_contents[title] = contents

print(len(title_contents))
with open('wiki_data/wiki_5k.json', 'w') as f:
    json.dump(title_contents, f)
fn.close()
