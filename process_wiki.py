from sklearn.feature_extraction.text import CountVectorizer
import json, networkx as nx, sys
from tqdm import tqdm
from gensim.models import KeyedVectors as KV
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

def load_features(data, label_dic, label_onehot, alpha):
    eval_set_num = '05'
    infile = 'wiki_data/updated_embedding_alpha_'
    if alpha <= 1:
        embeddings = KV.load_word2vec_format(infile + str(alpha)+'_'+eval_set_num, binary=True)
    else:
        em1 = KV.load_word2vec_format(infile + str(0.0)+'_'+eval_set_num, binary=True)
        em2 = KV.load_word2vec_format(infile + str(1.0)+'_'+eval_set_num, binary=True)
        embeddings = KV(vector_size = em1.vector_size*2)
        for t in tqdm(list(em1.vocab)):
            embeddings[t] = np.hstack((em1[t], em2[t]))

    ally_ty = []
    allx_tx = []
    allx_tx_dic = {}
    i = 0
    tmp_dic = {}
    print('appending training data...')
    dummy_dim = 800
    for k in tqdm(data):
        if k not in label_dic:
            continue
        tmp_vec = np.zeros(dummy_dim)
        tmp_vec[:embeddings.vector_size] = embeddings[k]
        allx_tx.append(tmp_vec)
        allx_tx_dic[k] = 0
        tmp_dic[k] = i
        one_hot = label_onehot[label_dic[k]]
        ally_ty.append(one_hot)
        i += 1

    nb_trains = i-1
    print('appending unlabeled data...')
    for k in tqdm(data):
        if k in label_dic:
            continue
        tmp_vec = np.zeros(dummy_dim)
        tmp_vec[:embeddings.vector_size] = embeddings[k]
        allx_tx.append(tmp_vec)
        allx_tx_dic[k] = 0
        tmp_dic[k] = i
        ally_ty.append(np.zeros(len(ally_ty[0])))
        i += 1

    for k in allx_tx_dic:
        allx_tx_dic[k] = allx_tx[tmp_dic[k]]

    return sp.sparse.csr_matrix(allx_tx), np.array(ally_ty), allx_tx_dic, nb_trains

def make_BOW(data, label_dic, label_onehot):
    vectorizer = CountVectorizer()
    print('constructing bag of words...')
    vectorizer.fit_transform(list(data.values()))
    ally_ty = []
    allx_tx = []
    allx_tx_dic = {}
    i = 0
    tmp_dic = {}
    print('appending training data...')
    for k in tqdm(data):
        if k not in label_dic:
            continue
        tmp_vec = vectorizer.transform([data[k]]).toarray()
        allx_tx.append(tmp_vec)
        allx_tx_dic[k] = 0
        tmp_dic[k] = i
        one_hot = label_onehot[label_dic[k]]
        ally_ty.append(one_hot)
        i += 1

    nb_trains = i-1
    print('appending unlabeled data...')
    for k in tqdm(data):
        if k in label_dic:
            continue
        tmp_vec = vectorizer.transform([data[k]]).toarray()
        allx_tx.append(tmp_vec)
        allx_tx_dic[k] = 0
        tmp_dic[k] = i
        ally_ty.append(np.zeros(len(ally_ty[0])))
        i += 1

    print('Reducing dimensionality with Truncated SVD...')
    tic()
    allx_tx = sp.sparse.csr_matrix(np.matrix(np.array(allx_tx)))
    allx_tx = svd(allx_tx, dim=2000, n_iter=1)
    toc()
    for k in allx_tx_dic:
        allx_tx_dic[k] = allx_tx[tmp_dic[k]]

    return sp.sparse.csr_matrix(allx_tx), np.array(ally_ty), allx_tx_dic, nb_trains

def split_train_test(allx_tx, ally_ty, label_dic, allx_tx_dic, nb_trainables, train_ratio, ti, label_onehot, edgefile):
    data = []
    for t in allx_tx_dic:
        if t in label_dic:
            data.append([allx_tx_dic[t], label_dic[t], t])
        else:
            data.append([allx_tx_dic[t], '_', t])

    new_ti = map_t_i(data, ti)
    tmp_graph = nx.to_dict_of_lists(get_CC(make_graph(edgefile, new_ti)))
    connected_nodes = list(tmp_graph.keys())

    data = []
    for t in allx_tx_dic:
        if ti[t] not in connected_nodes:
            continue
        if t in label_dic:
            data.append([allx_tx_dic[t], label_dic[t], t])
        else:
            data.append([allx_tx_dic[t], '_', t])

    data = np.array(data)
    np.random.shuffle(data)
    new_ti = map_t_i(data, ti)

    test_array = []
    labeled = np.where(data[:,1] != '_')[0]
    train_index = labeled[:int(len(labeled)*train_ratio)]
    test_index = labeled[int(len(labeled)*train_ratio):]

    train_x = [np.array(x).reshape(len(x)) for x in data[train_index][:, 0]]
    train_x = np.array(train_x)
    train_x = train_x.reshape(train_x.shape)
    train_x = sp.sparse.csr_matrix(np.matrix(train_x))

    train_y = [label_onehot[x] for x in data[train_index][:, 1]]
    train_y = np.array(train_y)

    new_allx = [np.array(x).reshape(len(x)) for x in data[:, 0]]
    new_allx = np.array(new_allx)
    new_allx = new_allx.reshape(new_allx.shape)
    new_allx = sp.sparse.csr_matrix(np.matrix(new_allx))

    new_ally = []
    for k in data:
        if k[1] == '_':
            new_ally.append(np.zeros(train_y.shape[1]))
        else:
            new_ally.append(label_onehot[k[1]])

    new_ally = np.array(new_ally)
    test_x = [np.array(x).reshape(len(x)) for x in data[test_index][:, 0]]
    test_x = np.array(test_x)
    test_x = test_x.reshape(test_x.shape)
    test_x = sp.sparse.csr_matrix(np.matrix(test_x))

    test_y = [label_onehot[x] for x in data[test_index][:, 1]]
    test_y = np.array(test_y)

    test_index = [str(x) for x in test_index]
    test_index  = ('\n').join(test_index)
    train_index = [str(x) for x in train_index]
    train_index  = ('\n').join(train_index)

    return new_allx, new_ally, train_x, train_y, test_x, test_y, test_index, train_index, new_ti

def map_t_i(data, old_ti):
    new_ti = {}
    for i in range(len(data)):
        new_ti[ old_ti[ data[i][2] ] ] = i

    return new_ti

def make_graph(edgefile, new_ti):
    g = nx.read_edgelist(edgefile, create_using=nx.Graph())
    x_nodes = [str(x) for x in list(new_ti.keys())]
    new_graph = g.subgraph(x_nodes)
    return new_graph

def construct_graph(edgefile, new_ti):
    g = nx.read_edgelist(edgefile, create_using=nx.Graph())
    keys = list(new_ti.keys())
    edges = list(g.edges())
    adj = np.zeros((len(new_ti), len(new_ti)))
    for e in tqdm(edges):
        if e[0] in keys and e[1] in keys:
            adj[int(new_ti[e[0]])][int(new_ti[e[1]])] = 1

    adj = np.matrix(adj)
    new_g = nx.Graph(nx.from_numpy_matrix(adj))
    return nx.to_dict_of_lists(new_g)

def svd(feature, dim=3000, n_iter=2):
    svd = TruncatedSVD(n_components=dim, n_iter=n_iter, random_state=42)
    return svd.fit_transform(feature)

def get_CC(graph):
    return graph.subgraph(max(nx.connected_components(graph)))

def main():
    sample_size = '5'
    alpha = float(sys.argv[2])
    ti_file = 'wiki_data/title_index.json'
    with open(ti_file, 'r', encoding='UTF8') as f:
        ti = json.loads(f.read())

    json_file = 'wiki_data/wiki_'+sample_size+'k.json'
    with open(json_file, 'r', encoding='UTF8') as f:
        wiki = json.loads(f.read())

    label_file = 'wiki_data/title_labels_'+sample_size+'k.json'
    with open(label_file, 'r') as g:
        label_dic = json.loads(g.read())
        print('number of labels:', len(set(label_dic.values())))

    nodes = np.load('wiki_data/wiki_node_'+sample_size+'k.npy')
    edgefile = 'wiki_data/edges_'+sample_size+'k.txt'
    cm_idx = np.load('wiki_data/cm_idx_05.npy')
    data = {}
    for title in wiki:
        if title not in cm_idx:
            continue
        data[title] = wiki[title]

    label_onehot = make_one_hot(label_dic)

    # allx_tx, ally_ty, allx_tx_dic, nb_trainables = make_BOW(data, label_dic, label_onehot)
    allx_tx, ally_ty, allx_tx_dic, nb_trainables = load_features(data, label_dic, label_onehot, alpha)
    print('number of trainables:', nb_trainables)
    train_ratio = float(sys.argv[1])
    print('spliting train and test....')
    allx, ally, x, y, tx, ty, test_index, train_index, new_ti = split_train_test(allx_tx, ally_ty, label_dic, allx_tx_dic, nb_trainables, train_ratio, ti, label_onehot, edgefile)

    print('making graph dictionary...')
    graph = construct_graph(edgefile, new_ti)

    with open('wiki_data/ind.wiki.allx', 'wb') as f:
        pkl.dump(allx, f)
    with open('wiki_data/ind.wiki.ally', 'wb') as f:
        pkl.dump(ally, f)
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
    with open('wiki_data/ind.wiki.train.index', 'w') as f:
        f.write(train_index)

if __name__ == '__main__':
    main()

#
