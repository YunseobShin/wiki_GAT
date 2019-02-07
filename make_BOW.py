from sklearn.feature_extraction.text import CountVectorizer
import json
from tqdm import tqdm
import scipy as sp, numpy as np
import pickle as pkl

def make_corpus(contents):
    return ' '.join(contents)

json_file = 'wiki.json'
with open(json_file, 'r', encoding='UTF8') as f:
    data = json.loads(f.read())

label_file = 'labels.json'
with open(label_file, 'r') as g:
    tl = json.loads(g.read())

vectorizer = CountVectorizer()
print('constructing bag of words...')
vectorizer.fit_transform(list(data.values()))

bag_of_words = []
labels = []
for k in tqdm(data):
    if k in tl:
        bag_of_words.append(vectorizer.transform([data[k]]).toarray())
        labels.append(tl[k])

bag_of_words = np.array(bag_of_words)
bag_of_words = np.matrix(bag_of_words)
bag_of_words = sp.sparse.csr_matrix(bag_of_words)



























#
