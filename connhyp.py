#!/usr/bin/env python

import sys
import os
from tqdm import tqdm
from gensim.models import word2vec
import numpy as np
from random import random
from sklearn.svm import SVC
import logging as log
log.basicConfig(level=log.INFO, format="%(asctime)s %(message)s")

# dimensions of the word embeddings
D = 100

# command line arguments
corpus1 = sys.argv[1] # preprocessed corpus file, first half
corpus2 = sys.argv[2] # preprocessed corpus file, second half
label1 = sys.argv[3] # labels of the connotative spectrum
label2 = sys.argv[4]
targetwordfile =  sys.argv[5] # filename containing the target words, one per line

# reading the target words
with open(targetwordfile) as f:
    targetwords = [line.strip().lower() for line in f]

# reading the two corpora
sentences1 = []
with open(corpus1) as f:
    for line in tqdm(f):
        words = line.strip().split(" ")
        sentences1.append(words)
sentences2 = []
with open(corpus2) as f:
    for line in tqdm(f):
        words = line.strip().split(" ")
        sentences2.append(words)

# appending the labels to the target words in the corpora 
sentences = []
for sentence in tqdm(sentences1):
    sentences.append(["{0}_{1}".format(word, label1) if (word in targetwords and random() < 0.66) else word for word in sentence])
for sentence in tqdm(sentences2):
    sentences.append(["{0}_{1}".format(word, label2) if (word in targetwords and random() < 0.66) else word for word in sentence])

# Word3vec
log.info("w2v fit")
w2v = word2vec.Word2Vec(sentences, vector_size=D) # for parallel CPU computing, use workers=...

# Connotative hyperplane training (SVM)
log.info("SVM")
X = []
y = []
with open("seed_{0}.txt".format(label1)) as f:
    for line in f:
        word = line.strip()
        if word in w2v.wv:
            X.append(w2v.wv[word])
            y.append(label1)
with open("seed_{0}.txt".format(label2)) as f:
    for line in f:
        word = line.strip()
        if word in w2v.wv:
            X.append(w2v.wv[word])
            y.append(label2)
X = np.array(X)
y = np.array(y)
classifier = SVC(kernel='linear')
classifier.fit(X,y)

# prediction
for targetword in targetwords:
    if not ("{0}_{1}".format(targetword, label1) in w2v.wv and "{0}_{1}".format(targetword, label2) in w2v.wv):
        continue
    target1 = w2v.wv["{0}_{1}".format(targetword, label1)]
    target2 = w2v.wv["{0}_{1}".format(targetword, label2)]
    X_test = np.array([target1, target2])
    norm = np.linalg.norm(classifier.coef_)
    d1, d2 = list(classifier.decision_function(X_test))
    print ("{0}\t{1}\t{2}".format(targetword, d1, d2).replace(".", ","))



