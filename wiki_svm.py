import numpy as np
from sklearn.svm import SVC
import sys, json, pickle as pkl
from tqdm import tqdm
from time_check import tic
from time_check import toc

def onehot_to_one(ys):
    new_ys = []
    for y in ys:
        ind = str(np.where(y==1)[0][0])
        new_ys.append(ind)
    return np.array(new_ys)

def training_SVM(train_x, train_y):
    clf = SVC(gamma='auto', decision_function_shape='ovo', kernel='rbf')
    print('Traning SVM...')
    clf.fit(train_x, train_y)
    return clf


with open('wiki_data/ind.wiki.x', 'rb') as f:
    train_x = pkl.load(f, encoding='latin1')
with open('wiki_data/ind.wiki.y', 'rb') as f:
    train_y = pkl.load(f, encoding='latin1')
    train_y = onehot_to_one(train_y)
with open('wiki_data/ind.wiki.tx', 'rb') as f:
    test_x = pkl.load(f, encoding='latin1')
with open('wiki_data/ind.wiki.ty', 'rb') as f:
    test_y = pkl.load(f, encoding='latin1')
    test_y = onehot_to_one(test_y)

tic()
model = training_SVM(train_x, train_y)
tic()

acc = model.score(test_x, test_y)
print('Testing Acc: '+str(acc))
