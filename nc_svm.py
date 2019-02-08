import numpy as np
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import average_precision_score
import sys, json
from gensim.models import KeyedVectors
from tqdm import tqdm
from time_check import tic
from time_check import toc

use_tfidf = 0
eval_set_num = '02'
alpha = sys.argv[1]
if use_tfidf == 0:
    infile = 'embeddings/updated_embedding_alpha_'
else:
    infile = 'embeddings/updated_embedding_tfidf_alpha_'
embeddings = KeyedVectors.load_word2vec_format(infile + str(alpha)+'_'+eval_set_num, binary=True)

fname = 'title_labels_'+eval_set_num+'.json'
f = open('title_index.json').read()
t_i = json.loads(f)
f = open('index_title.json').read()
i_t = json.loads(f)
with open(fname, 'r') as g:
    t_l = json.loads(g.read())
common_idx = np.load('cm_idx_'+eval_set_num+'.npy')
data = []
for t in t_l:
    if t not in common_idx:
        continue
    data.append([embeddings[t], t_l[t]])

data = np.array(data)

labels = set(list(t_l.values()))
data_labels = []
for label in labels:
    data_labels.append(data[np.where(data[:,1]==label)])

data_labels = np.array(data_labels)

train = []
test = []

for data_label in data_labels:
    train.append(data_label[:int(len(data_label)*0.8)])
    test.append(data_label[int(len(data_label)*0.8):])

train_x = []
train_y = []
test_x = []
test_y = []
for i in range(len(train)):
    if train[i].shape[0] < 10:
        continue
    for t in train[i]:
        train_x.append(t[0])
        train_y.append(t[1])
    for t in test[i]:
        test_x.append(t[0])
        test_y.append(t[1])

np.save('./features/train_x_'+str(alpha), train_x)
np.save('./features/train_y_'+str(alpha), train_y)

label_index = {}
i=0
for data_label in data_labels:
    if len(data_label) < 1:
        # print(data_label)
        continue
    label = data_label[0][1]
    if label not in set(test_y):
        continue
    else:
        label_index[label] = i
        i += 1

q=[]
for ty in test_y:
    q.append(label_index[ty])
test_y = q

q=[]
for ty in train_y:
    q.append(label_index[ty])

train_y = [float(x) for x in q]

print('training data size:', len(train_y))
print('the number of classes:', len(set(label_index)))

def training_SVM(train_x, train_y):
    clf = SVC(gamma='auto', decision_function_shape='ovo', kernel='rbf')
    print('Traning SVM...')
    clf.fit(train_x, train_y)
    return clf

def training_LR(train_x, train_y):
    model = LR(solver='saga')
    print('Traning Linear Regression...')
    # tic()
    model.fit(train_x, train_y)
    # toc()
    return model

model = training_SVM(train_x, train_y)
# model = training_LR(train_x, train_y)

acc = model.score(test_x, test_y)
print('Testing Acc alpha:'+str(alpha)+':'+str(acc))
# preds = model.decision_function(test_x)
# AP = average_precision_score(test_y, preds)
# AP = np.mean(precisions)
# print('Testing AP:', AP)
# print('Testing AP:', AP)
