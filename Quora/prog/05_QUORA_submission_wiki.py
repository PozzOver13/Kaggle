# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 08:24:57 2019

@author: CG08900
"""

# > INITIALIZATION

# >> imports

import gensim as gs
import sklearn.feature_extraction as fe
import sklearn.model_selection as ms
import sklearn.metrics as met
import pandas as pd
import numpy as np
import os
import gc
import time
import re

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers import Activation, CuDNNGRU, Conv1D, GRU
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

#import os
#print(os.listdir("../input"))

# >> constant values

EMBEDDING_FILE_GLOVE = '../input/embeddings/paragram.840B.300d/glove.840B.300d.txt'
EMBEDDING_FILE_GOOGLE = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
EMBEDDING_FILE_PARAGRAM = 'datainput/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
EMBEDDING_FILE_WIKI = 'datainput/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
TOKEN_FILTERS = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'
EMBED_SIZE = 300


# > LOAD DATA
os.chdir('C:/Users/cg08900/Documents/Pandora/Personale/kaggle/Quora/')
train_df = pd.read_csv("datainput/train.csv")
test_df = pd.read_csv("datainput/test.csv")
print("Train shape : ", train_df.shape)
print("Test shape : ", test_df.shape)

print("Columns:")
print(train_df.dtypes)

# > T/V SPLIT

# >> split between training and validation
train_df, val_df = ms.train_test_split(train_df, test_size=0.15, 
                                       random_state=2019)

print(freq(train_df, "target"))
print(freq(val_df, "target"))


# > TEXT CLEANING

# >> prepare corpus

train_corpus = train_df['question_text']
val_corpus = val_df['question_text']
test_corpus = test_df['question_text']

# >> common to all embeddings

train_corpus = clean_corpus(train_corpus)
val_corpus = clean_corpus(val_corpus)

# >> misspelled and special

train_corpus_cl = clean_misspelled_and_special(train_corpus)
val_corpus_cl = clean_misspelled_and_special(val_corpus)

# create vocabulary for train sentences
train_vocabulary_wiki = \
    create_word_count(train_corpus_cl, to_lowercase=True, add_to_filters="'’”")
   
# > EMBEDDINGS

# >> wiki

embedding_wiki_filtered = \
    read_and_filter_embedding(EMBEDDING_FILE_WIKI,
                              train_vocabulary_wiki,
                              source = "wiki")

# check % of words found in embedding
check_coverage(train_vocabulary_wiki, embedding_wiki_filtered)

# > NN TRAINING (wiki)

sentences_maxlen = 150
max_features = 80000

# >> data preparation

train_kit_wiki = fit_tokenizer(train_corpus_cl, 
                                embedding_wiki_filtered,
                                max_features=max_features,
                                maxlen=sentences_maxlen,
                                add_filters="'’”")

train_X = train_kit_wiki['train_matrix']

val_X = apply_tokenizer(train_kit_wiki['tokenizer'], 
                        val_corpus_cl, sentences_maxlen)

test_X = apply_tokenizer(train_kit_wiki['tokenizer'], 
                         test_corpus, sentences_maxlen)

train_y = train_df['target'].values
val_y = val_df['target'].values

# >> NN training

inp = Input(shape=(sentences_maxlen,), batch_shape=(512, sentences_maxlen))
x = Embedding(max_features, EMBED_SIZE, 
              weights=[train_kit_wiki['embedding_matrix']])(inp)
x = Bidirectional(CuDNNGRU(64, return_sequences=True, stateful=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model_nn_wiki = Model(inputs=inp, outputs=x)
model_nn_wiki.compile(loss='binary_crossentropy', 
                       optimizer='adam', metrics=['accuracy'])
print(model_nn_wiki.summary())

# NOTE: in order to use stateful LSTM NN you have to extract
#       samples with a number of observation which has to be a
#       multiple of the batch size
model_nn_wiki.fit(train_X[0:1110016,:], train_y[0:1110016],
                   batch_size=512, epochs=2,
                   validation_data=(val_X[0:512*382, :], val_y[0:512*382]))