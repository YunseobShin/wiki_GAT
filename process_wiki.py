from sklearn.feature_extraction.text import CountVectorizer
import json, networkx as nx
from tqdm import tqdm
import scipy as sp, numpy as np
import pickle as pkl

def make_one_hot(label_dic):
    labels = list(set(label_dic.values()))
    l_one = {}
    k = len(labels)
    for i in range(len(labels)):
        l_one[labels[i]] = np.zeros(k)
        l_one[labels[i]][i] = 1

    return l_one

def make_BOW(data, label_dic):
    vectorizer = CountVectorizer()
    print('constructing bag of words...')
    vectorizer.fit_transform(list(data.values()))
    ally = []
    label_onehot = make_one_hot(label_dic)
    allx = []
    for k in tqdm(data):
        if k in label_dic:
            allx.append(vectorizer.transform([data[k]]).toarray())
            ally.append(label_onehot[label_dic[k]])

    allx = np.array(allx)
    allx = np.matrix(allx)
    return sp.sparse.csr_matrix(allx), np.array(ally)

def split_train_test(data, label_dic):
    pass

def make_graph():

def main():
    json_file = 'wiki_data/wiki_30k.json'
    with open(json_file, 'r', encoding='UTF8') as f:
        wiki = json.loads(f.read())

    label_file = 'wiki_data/title_labels_30k.json'
    with open(label_file, 'r') as g:
        label_dic = json.loads(g.read())

    nodes = np.load('wiki_data/wiki_node_30k.npy')
    edges = open('wiki_data/edges_30k.txt').read()

    data = {}
    for title in wiki:
        if title in label_dic:
            data[title] = wiki[title]

    allx, ally = make_BOW(data, label_dic)




    allx_file = open('wiki_data/ind.wiki.allx', 'wb')
    pkl.dump(allx, allx_file)


















if __name__ == '__main__':
    main()

#
