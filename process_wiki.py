from sklearn.feature_extraction.text import CountVectorizer
import json, networkx as nx
from tqdm import tqdm
import scipy as sp, numpy as np
import pickle as pkl
from sklearn.decomposition import TruncatedSVD
from time_check import tic
from time_check import toc

def make_one_hot(label_dic):
    labels = list(set(label_dic.values()))
    l_one = {}
    k = len(labels)
    for i in range(len(labels)):
        l_one[labels[i]] = np.zeros(k)
        l_one[labels[i]][i] = 1

    return l_one

def make_BOW(data, label_dic, label_onehot):
    vectorizer = CountVectorizer()
    print('constructing bag of words...')
    vectorizer.fit_transform(list(data.values()))
    ally_ty = []
    allx_tx = []
    allx_tx_dic = {}
    i = 0
    tmp_dic = {}
    for k in tqdm(data):
        if k in label_dic:
            tmp_vec = vectorizer.transform([data[k]]).toarray()
            allx_tx.append(tmp_vec)
            allx_tx_dic[k] = 0
            tmp_dic[k] = i
            one_hot = label_onehot[label_dic[k]]
            ally_ty.append(one_hot)
            i += 1

    print('Reducing dimensionality with Truncated SVD...')
    tic()
    allx_tx = sp.sparse.csr_matrix(np.matrix(np.array(allx_tx)))
    allx_tx = svd(allx_tx)
    toc()
    for k in allx_tx_dic:
        allx_tx_dic[k] = allx_tx[tmp_dic[k]]

    return sp.sparse.csr_matrix(allx_tx), np.array(ally_ty), allx_tx_dic

def split_train_test(label_dic, allx_tx_dic, train_ratio, ti, label_onehot):
    common_idx = np.load('wiki_data/cm_idx.npy')
    data = []
    for t in label_dic:
        if t not in common_idx:
            continue
        data.append([allx_tx_dic[t], label_dic[t], t])
    data = np.array(data)
    np.random.shuffle(data)
    new_ti = map_t_i(data, ti)

    train = []
    test = []
    test_array = []
    train_chunk = data[:int(len(data)*train_ratio)]
    test_chunk = data[int(len(data)*train_ratio):]
    test_index = range(int(len(data)*train_ratio), len(data))

    train_x = []
    test_x = []
    train_y = []
    test_y = []

    for tr in train_chunk:
        train_x.append(list(tr[0]))
        train_y.append(label_onehot[tr[1]])
    for ts in test_chunk:
        test_x.append(list(ts[0]))
        test_y.append(label_onehot[ts[1]])

    train_x = np.matrix(np.array(train_x))
    test_x = np.matrix(np.array(test_x))
    train_y = np.array(train_y)
    test_y = np.array(test_y)
    test_index = [str(x) for x in test_index]
    test_index  = ('\n').join(test_index)

    return sp.sparse.csr_matrix(train_x), train_y, sp.sparse.csr_matrix(test_x), test_y, test_index, new_ti


def map_t_i(data, old_ti):
    new_ti = {}
    for i in range(len(data)):
        new_ti[ old_ti[ data[i][2] ] ] = str(i)

    return new_ti

def make_graph(edgefile, new_ti):
    new_graph = nx.DiGraph()
    g = nx.read_edgelist(edgefile, create_using=nx.DiGraph())
    common_idx = np.load('wiki_data/cm_idx.npy')
    for e in tqdm(list(g.edges())):
        if e[0] not in new_ti or e[1] not in new_ti:
            continue
        new_graph.add_edge(new_ti[e[0]], new_ti[e[1]])
    return nx.to_dict_of_lists(new_graph)

def svd(feature):
    svd = TruncatedSVD(n_components=2048, n_iter=1, random_state=42)
    return svd.fit_transform(feature)

def main():
    ti_file = 'wiki_data/title_index.json'
    with open(ti_file, 'r', encoding='UTF8') as f:
        ti = json.loads(f.read())

    json_file = 'wiki_data/wiki_30k.json'
    with open(json_file, 'r', encoding='UTF8') as f:
        wiki = json.loads(f.read())

    label_file = 'wiki_data/title_labels_30k.json'
    with open(label_file, 'r') as g:
        label_dic = json.loads(g.read())
        print('number of labels:', len(set(label_dic.values())))

    nodes = np.load('wiki_data/wiki_node_30k.npy')
    edgefile = 'wiki_data/edges_30k.txt'


    data = {}
    common_idx = np.load('wiki_data/cm_idx.npy')
    for title in wiki:
        if title in label_dic and title in common_idx:
            data[title] = wiki[title]

    label_onehot = make_one_hot(label_dic)
    
    allx_tx, ally_ty, allx_tx__dic = make_BOW(data, label_dic, label_onehot)

    train_ratio = 0.8
    print('spliting train / test')
    x, y, tx, ty, test_index, new_ti = split_train_test(label_dic, allx_tx__dic, train_ratio, ti, label_onehot)
    print('making graph dictionary...')
    graph = make_graph(edgefile, new_ti)


    with open('wiki_data/ind.wiki.allx', 'wb') as f:
        pkl.dump(x, f)
    with open('wiki_data/ind.wiki.ally', 'wb') as f:
        pkl.dump(y, f)
    with open('wiki_data/ind.wiki.x', 'wb') as f:
        pkl.dump(x, f)
    with open('wiki_data/ind.wiki.y', 'wb') as f:
        pkl.dump(y, f)
    with open('wiki_data/ind.wiki.tx', 'wb') as f:
        pkl.dump(tx, f)
    with open('wiki_data/ind.wiki.ty', 'wb') as f:
        pkl.dump(ty, f)
    with open('wiki_data/ind.wiki.graph', 'wb') as f:
        pkl.dump(graph, f)
    with open('wiki_data/ind.wiki.test.index', 'w') as f:
        f.write(test_index)





if __name__ == '__main__':
    main()

#
