# > INITIALIZATION

import re
import gc
import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from pandas import read_csv
from gensim.models import KeyedVectors

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers

# >> constant values to change

MAX_FEATURES = 50000
MAX_SENTENCE_LEN = 150

# >> imports


# >> constant values NOT to change

EMBEDDING_FILE_GLOVE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
EMBEDDING_FILE_GOOGLE = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
EMBEDDING_FILE_PARAGRAM = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
EMBEDDING_FILE_WIKI = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
TOKEN_FILTERS = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'
EMBED_SIZE = 300

MISSPELLED_WORDS = {'colour': 'color',
                    'centre': 'center',
                    'favourite': 'favorite',
                    'travelling': 'traveling',
                    'counselling': 'counseling',
                    'theatre': 'theater',
                    'cancelled': 'canceled',
                    'labour': 'labor',
                    'organisation': 'organization',
                    'wwii': 'world war 2',
                    'citicise': 'criticize',
                    'youtu': 'youtube',
                    'youtubebe': 'youtube',
                    'Qoura': 'Quora',
                    'sallary': 'salary',
                    'Whta': 'What',
                    'narcisist': 'narcissist',
                    'howdo': 'how do',
                    'whatare': 'what are',
                    'howcan': 'how can',
                    'howmuch': 'how much',
                    'howmany': 'how many',
                    'whydo': 'why do',
                    'doI': 'do I',
                    'theBest': 'the best',
                    'howdoes': 'how does',
                    'mastrubation': 'masturbation',
                    'mastrubate': 'masturbate',
                    "mastrubating": 'masturbating',
                    'pennis': 'penis',
                    'Etherium': 'Ethereum',
                    'narcissit': 'narcissist',
                    'bigdata': 'big data',
                    '2k17': '2017', '2k18': '2018',
                    'qouta': 'quota',
                    'exboyfriend': 'ex boyfriend',
                    'airhostess': 'air hostess',
                    "whst": 'what',
                    'watsapp': 'whatsapp',
                    'demonitisation': 'demonetization',
                    'demonitization': 'demonetization',
                    'Demonetization': 'demonetization',
                    'demonetisation': 'demonetization',
                    'Quorans': 'users',
                    'quorans': 'users',
                    'Pokémon': 'Pokemon',
                    'pokémon': 'pokemon'}

PUNCT = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
PUNCT_MAPPING = {"‘": "'", "₹": "e", "´": "'", "°": "",
                 "€": "e", "™": "tm", "√": " sqrt ",
                 "×": "x", "²": "2", "—": "-", "–": "-",
                 "’": "'", "_": "-", "`": "'", '“': '"',
                 '”': '"', '“': '"', "£": "e", '∞': 'infinity',
                 'θ': 'theta', '÷': '/',
                 'α': 'alpha', '•': '.', 'à': 'a',
                 '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi'}


# >> FUNCTION DEFINITION

def clean_numbers(sentence):
    return (re.sub(r"\d{2,}(\.[0-9]+)*",
                   lambda x: len(x.group()) * "#",
                   sentence))


def correct_spelling(text, dic):
    for word in dic.keys():
        text = text.replace(word, dic[word])

    return text


def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])

    for p in punct:
        text = text.replace(p, f' {p} ')

    specials = {'\u200b': ' ', '…': ' ... ',
                '\ufeff': '', 'करना': '', 'है': ''}
    for s in specials:
        text = text.replace(s, specials[s])

    return text


def clean_corpus(corpus, to_lower=True):
    corpus = [correct_spelling(sentence, MISSPELLED_WORDS) for sentence in corpus]

    corpus = [clean_special_chars(sentence, PUNCT, PUNCT_MAPPING) for sentence in corpus]

    corpus = [clean_numbers(sentence) for sentence in corpus]

    if (to_lower):
        corpus = [sentence.lower() for sentence in corpus]

    return (corpus)


def fit_tokenizer(corpus, max_features=None, add_filters=None):
    # set tokenizer filters
    filtersForTokenizer = TOKEN_FILTERS
    if add_filters is not None:
        filtersForTokenizer = filtersForTokenizer + add_filters

    # tokenize the sentences
    if max_features is not None:
        tokenizer = Tokenizer(num_words=max_features,
                              filters=filtersForTokenizer)
    else:
        tokenizer = Tokenizer(filters=filtersForTokenizer)

    tokenizer.fit_on_texts(corpus)

    return (tokenizer)


def create_word_matrix(tokenizer, corpus, maxlen):
    tokenizedCorpus = pad_sequences(tokenizer.texts_to_sequences(corpus),
                                    maxlen=maxlen)
    return (tokenizedCorpus)


def load_glove(tokenizer, embedding_file):
    # load whole embedding file

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddingsIndex = dict(get_coefs(*o.split(" ")) for o in open(embedding_file))

    wholeEmbedding = np.stack(list(embeddingsIndex.values()))
    embeddingMean, embeddingStd = wholeEmbedding.mean(), wholeEmbedding.std()
    embeddingSize = wholeEmbedding.shape[1]

    # create embedding matrix where oov words are
    # initialized to a random normal vector

    wordIndex = tokenizer.word_index
    numberOfWords = min(tokenizer.num_words, len(wordIndex))
    embeddingMatrix = np.random.normal(embeddingMean, embeddingStd,
                                       (numberOfWords, embeddingSize))
    for word, i in wordIndex.items():
        if i >= numberOfWords: continue
        embeddingVector = embeddingsIndex.get(word)
        if embeddingVector is not None:
            embeddingMatrix[i] = embeddingVector

    # finally get out-of-vocabulary words

    oov = {word: n for word, n in tokenizer.word_counts.items() if word not in embeddingsIndex.keys()}

    oovPercentSingleWord = len(oov) / len(wordIndex)
    oovPercentAll = sum(oov.values()) / sum(tokenizer.word_counts.values())

    oovToPrint = pd.DataFrame(sorted(oov.items(), key=lambda kv: kv[1],
                                     reverse=True)[0:14],
                              columns=['Word', 'N'])

    printTemplate = """
    
    GLOVE - Percentage of words not in embedding: {0:.2%} ({2:.2%} of vocabulary).
    {1}
    
    Shape of final embedding matrix: {3}
    """
    print(printTemplate.format(oovPercentAll, oovToPrint,
                               oovPercentSingleWord, embeddingMatrix.shape))

    return embeddingMatrix


def load_wiki(tokenizer, embedding_file):
    # load whole embedding file

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddingsIndex = dict(get_coefs(*o.split(" ")) for o in open(embedding_file, encoding='latin')
                           if len(o) > 100)

    wholeEmbedding = np.stack(list(embeddingsIndex.values()))
    embeddingMean, embeddingStd = wholeEmbedding.mean(), wholeEmbedding.std()
    embeddingSize = wholeEmbedding.shape[1]

    # create embedding matrix where oov words are
    # initialized to a random normal vector

    wordIndex = tokenizer.word_index
    numberOfWords = min(tokenizer.num_words, len(wordIndex))
    embeddingMatrix = np.random.normal(embeddingMean, embeddingStd,
                                       (numberOfWords, embeddingSize))

    for word, i in wordIndex.items():
        if i >= numberOfWords: continue
        embeddingVector = embeddingsIndex.get(word)
        if embeddingVector is not None:
            embeddingMatrix[i] = embeddingVector

    # finally get out-of-vocabulary words

    oov = {word: n for word, n in tokenizer.word_counts.items() if word not in embeddingsIndex.keys()}

    oovPercentSingleWord = len(oov) / len(wordIndex)
    oovPercentAll = sum(oov.values()) / sum(tokenizer.word_counts.values())

    oovToPrint = pd.DataFrame(sorted(oov.items(), key=lambda kv: kv[1],
                                     reverse=True)[0:14],
                              columns=['Word', 'N'])

    printTemplate = """
    
    WIKI - Percentage of words not in embedding: {0:.2%} ({2:.2%} of vocabulary).
    {1}
    
    Shape of final embedding matrix: {3}
    """
    print(printTemplate.format(oovPercentAll, oovToPrint,
                               oovPercentSingleWord, embeddingMatrix.shape))

    return embeddingMatrix


def load_paragram(tokenizer, embedding_file):
    # load whole embedding file

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddingsIndex = dict(get_coefs(*o.split(" ")) for o in open(embedding_file, encoding='latin'))

    wholeEmbedding = np.stack(list(embeddingsIndex.values()))
    embeddingMean, embeddingStd = wholeEmbedding.mean(), wholeEmbedding.std()
    embeddingSize = wholeEmbedding.shape[1]

    # create embedding matrix where oov words are
    # initialized to a random normal vector

    wordIndex = tokenizer.word_index
    numberOfWords = min(tokenizer.num_words, len(wordIndex))
    embeddingMatrix = np.random.normal(embeddingMean, embeddingStd,
                                       (numberOfWords, embeddingSize))
    for word, i in wordIndex.items():
        if i >= numberOfWords: continue
        embeddingVector = embeddingsIndex.get(word)
        if embeddingVector is not None:
            embeddingMatrix[i] = embeddingVector

    # finally get out-of-vocabulary words

    oov = {word: n for word, n in tokenizer.word_counts.items() if word not in embeddingsIndex.keys()}

    oovPercentSingleWord = len(oov) / len(wordIndex)
    oovPercentAll = sum(oov.values()) / sum(tokenizer.word_counts.values())

    oovToPrint = pd.DataFrame(sorted(oov.items(), key=lambda kv: kv[1],
                                     reverse=True)[0:14],
                              columns=['Word', 'N'])

    printTemplate = """
    
    PARAGRAM - Percentage of words not in embedding: {0:.2%} ({2:.2%} of vocabulary).
    {1}
    
    Shape of final embedding matrix: {3}
    """
    print(printTemplate.format(oovPercentAll, oovToPrint,
                               oovPercentSingleWord, embeddingMatrix.shape))

    return embeddingMatrix


def load_google(tokenizer, embedding_file):
    # load whole embedding file

    embeddingsKVector = KeyedVectors.load_word2vec_format(embedding_file, binary=True)

    wholeEmbedding = embeddingsKVector.vectors
    embeddingMean, embeddingStd = wholeEmbedding.mean(), wholeEmbedding.std()
    embeddingSize = wholeEmbedding.shape[1]
    wholeEmbedding = None

    # create embedding matrix where oov words are
    # initialized to a random normal vector

    wordIndex = tokenizer.word_index
    numberOfWords = min(tokenizer.num_words, len(wordIndex))
    embeddingMatrix = np.random.normal(embeddingMean, embeddingStd,
                                       (numberOfWords, embeddingSize))
    embeddingWords = embeddingsKVector.vocab.keys()

    for word, i in wordIndex.items():
        if i >= numberOfWords: continue
        if word in embeddingWords:
            embeddingVector = embeddingsKVector.get_vector(word)
        else:
            embeddinVector = None
        if embeddingVector is not None:
            embeddingMatrix[i] = embeddingVector

    # finally get out-of-vocabulary words

    oov = {word: n for word, n in tokenizer.word_counts.items() if word not in embeddingWords}

    oovPercentSingleWord = len(oov) / len(wordIndex)
    oovPercentAll = sum(oov.values()) / sum(tokenizer.word_counts.values())

    oovToPrint = pd.DataFrame(sorted(oov.items(), key=lambda kv: kv[1],
                                     reverse=True)[0:14],
                              columns=['Word', 'N'])

    printTemplate = """
    
    GOOGLE - Percentage of words not in embedding: {0:.2%} ({2:.2%} of vocabulary).
    {1}
    
    Shape of final embedding matrix: {3}
    """
    print(printTemplate.format(oovPercentAll, oovToPrint,
                               oovPercentSingleWord, embeddingMatrix.shape))

    # clean RAM memory, just in case
    embeddingsKVector = None

    return embeddingMatrix


# borrowed from:
# https://www.kaggle.com/suicaokhoailang/lstm-attention-baseline-0-652-lb
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


def best_threshold(y_hat, y):
    # borrowed from: 
    # www.kaggle.com/jannen/reaching-0-7-fork-from-bilstm-attention-kfold
    tmp = [0, 0, 0]  # idx, cur, max
    delta = 0
    for tmp[0] in np.arange(0.1, 0.501, 0.01):
        tmp[1] = f1_score(y, (y_hat > tmp[0]).astype(int))
        if tmp[1] > tmp[2]:
            delta = tmp[0]
            tmp[2] = tmp[1]
    print('Best threshold is {:.4f} with F1 score: {:.4f}'.format(delta, tmp[2]))
    return delta, tmp[2]


def nn_fit_test(model, batch_size=512, epochs=2, cl_weight=None,
                datasets=None):
    # recover data
    train_X = datasets['train_X']
    train_y = datasets['train_y']
    val_X = datasets['val_X']
    val_y = datasets['val_y']
    test_X = datasets['test_X']

    # model fitting
    if cl_weight is None:
        model.fit(train_X, train_y, batch_size=batch_size, epochs=epochs,
                  validation_data=(val_X, val_y))
    else:
        model.fit(train_X, train_y, batch_size=batch_size, epochs=epochs,
                  validation_data=(val_X, val_y), class_weight=cl_weight)

        # apply model to training, validation and test
    train_y_hat = model.predict([train_X], batch_size=batch_size, verbose=1)
    val_y_hat = model.predict([val_X], batch_size=batch_size, verbose=1)
    test_y_hat = model.predict([test_X], batch_size=batch_size, verbose=1)

    # F1 score on validation
    bt, f1 = best_threshold(val_y_hat, val_y)

    return train_y_hat, val_y_hat, test_y_hat, bt, f1


def add_features(df0):

    df = df0.copy()
    df['question_text'] = df['question_text'].apply(lambda x: str(x))
    df['total_length'] = df['question_text'].apply(len)
    df['capitals'] = df['question_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    df['caps_vs_length'] = df.apply(lambda row: float(row['capitals']) / float(row['total_length']),
                                    axis=1)
    df['num_words'] = df.question_text.str.count('\S+')
    df['num_unique_words'] = df['question_text'].apply(lambda comment: len(set(w for w in comment.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words']

    return df


# > LOAD DATA

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


# > TV SPLIT + DATA CLEANING

# >> training / validation split

train_df, val_df = train_test_split(train_df, test_size=0.15,
                                    random_state=2019)

# >> extract corpus

train_corpus = train_df['question_text']
val_corpus = val_df['question_text']
test_corpus = test_df['question_text']

# >> clean corpus

train_corpus = clean_corpus(train_corpus)
val_corpus = clean_corpus(val_corpus)
test_corpus = clean_corpus(test_corpus)

# > TOKENIZE

# >> fit tokenizer

train_tokenizer = fit_tokenizer(train_corpus, max_features=MAX_FEATURES,
                                add_filters="'’")


# > LOAD EMBEDDINGS

embedding_glove_filtered = load_glove(train_tokenizer, EMBEDDING_FILE_GLOVE)
gc.collect()  # probably useless
embedding_paragram_filtered = load_paragram(train_tokenizer, EMBEDDING_FILE_PARAGRAM)
gc.collect()
embedding_wiki_filtered = load_wiki(train_tokenizer, EMBEDDING_FILE_WIKI)
gc.collect()


# > FINAL STEPS BEFORE NN

# >> apply tokenizer and create matrices

train_nn_x = create_word_matrix(train_tokenizer,
                                train_corpus, MAX_SENTENCE_LEN)
val_nn_x = create_word_matrix(train_tokenizer,
                              val_corpus, MAX_SENTENCE_LEN)
test_nn_x = create_word_matrix(train_tokenizer,
                               test_corpus, MAX_SENTENCE_LEN)

# >> extract target array

train_nn_y = train_df['target']
val_nn_y = val_df['target']


# DATA IN & OUT (run just once!)

# >> input data

data_4_nn = {'train_X': train_nn_x,
             'train_y': train_nn_y,
             'val_X': val_nn_x,
             'val_y': val_nn_y,
             'test_X': test_nn_x}

# >> initialize results

nn_fit_results = dict()


# > MODEL FUNCTIONS

def model_lstm_attention(embedding_matrix, sentences_maxlen=MAX_SENTENCE_LEN,
                         max_features=MAX_FEATURES, embedding_size=EMBED_SIZE):
    if max_features != embedding_matrix.shape[0]:
        max_features = embedding_matrix.shape[0]
    inp = Input(shape=(sentences_maxlen,))
    x = Embedding(max_features, embedding_size,
                  weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    x = Dropout(0.10)(x)
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    x = Attention(sentences_maxlen)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.15)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return model


def model_gru_pooling(embedding_matrix, sentences_maxlen=MAX_SENTENCE_LEN,
                      max_features=MAX_FEATURES, embedding_size=EMBED_SIZE):
    if max_features != embedding_matrix.shape[0]:
        max_features = embedding_matrix.shape[0]                          
    inp = Input(shape=(sentences_maxlen,))
    x = Embedding(max_features, embedding_size,
                  weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True, stateful=False))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    x = Dense(64, activation="relu")(conc)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return model


# > NN PT.1: LSTM W/ATTENTION (ONE EMBEDDING AT A TIME)

# Based on:
# https://www.kaggle.com/shujian/mix-of-nn-models-based-on-meta-embedding

# >> Glove only

mod_lstmA_glove = model_lstm_attention(embedding_glove_filtered)
print(mod_lstmA_glove.summary())

nn_fit_results['lstm a glove only'] = nn_fit_test(mod_lstmA_glove, epochs=3, datasets=data_4_nn)

# >> Wiki only

mod_lstmA_wiki = model_lstm_attention(embedding_wiki_filtered)
print(mod_lstmA_wiki.summary())

nn_fit_results['lstm a wiki only'] = nn_fit_test(mod_lstmA_wiki, epochs=3, datasets=data_4_nn)

# >> Paragram only

mod_lstmA_paragram = model_lstm_attention(embedding_paragram_filtered)
print(mod_lstmA_paragram.summary())

nn_fit_results['lstm a paragram only'] = nn_fit_test(mod_lstmA_paragram, epochs=3, datasets=data_4_nn)


# # > NN PT.2: GRU W/POOLING
# 
# # >> Glove only
# 
# mod_gruP_glove = model_gru_pooling(embedding_glove_filtered)
# print(mod_gruP_glove.summary())
# 
# nn_fit_results['gru pool glove only'] = nn_fit_test(mod_gruP_glove, epochs=3, datasets=data_4_nn)
# 
# # >> Wiki only
# 
# mod_gruP_wiki = model_gru_pooling(embedding_wiki_filtered)
# print(mod_gruP_wiki.summary())
# 
# nn_fit_results['gru pool wiki only'] = nn_fit_test(mod_gruP_wiki, epochs=3, datasets=data_4_nn)
# 
# # >> Paragram only
# 
# mod_gruP_paragram = model_gru_pooling(embedding_paragram_filtered)
# print(mod_gruP_paragram.summary())
# 
# nn_fit_results['gru pool paragram only'] = nn_fit_test(mod_gruP_paragram, epochs=3, datasets=data_4_nn)
# 

# > FINAL PREDICTION

# >> assign probability with embeddings

# train
train_df['pr_lstma_glove'] = nn_fit_results['lstm a glove only'][0]
train_df['pr_lstma_wiki'] = nn_fit_results['lstm a wiki only'][0]
train_df['pr_lstma_paragram'] = nn_fit_results['lstm a paragram only'][0]
#train_df['pr_grup_glove'] = nn_fit_results['gru pool glove only'][0]
#train_df['pr_grup_wiki'] = nn_fit_results['gru pool wiki only'][0]
#train_df['pr_grup_paragram'] = nn_fit_results['gru pool paragram only'][0]

# validation
val_df['pr_lstma_glove'] = nn_fit_results['lstm a glove only'][1]
val_df['pr_lstma_wiki'] = nn_fit_results['lstm a wiki only'][1]
val_df['pr_lstma_paragram'] = nn_fit_results['lstm a paragram only'][1]
# val_df['pr_grup_glove'] = nn_fit_results['gru pool glove only'][1]
# val_df['pr_grup_wiki'] = nn_fit_results['gru pool wiki only'][1]
# val_df['pr_grup_paragram'] = nn_fit_results['gru pool paragram only'][1]

# test
test_df['pr_lstma_glove'] = nn_fit_results['lstm a glove only'][2]
test_df['pr_lstma_wiki'] = nn_fit_results['lstm a wiki only'][2]
test_df['pr_lstma_paragram'] = nn_fit_results['lstm a paragram only'][2]
# test_df['pr_grup_glove'] = nn_fit_results['gru pool glove only'][2]
# test_df['pr_grup_wiki'] = nn_fit_results['gru pool wiki only'][2]
# test_df['pr_grup_paragram'] = nn_fit_results['gru pool paragram only'][2]

# >> final probability and threshold

pr_mean_val = (val_df['pr_lstma_glove'] + val_df['pr_lstma_paragram'] + \
           val_df['pr_lstma_wiki']) / 3

bt, f1 = best_threshold(pr_mean_val, val_nn_y)


# > SAVE SUBMISSION

pr_mean_test = (test_df['pr_lstma_glove'] + test_df['pr_lstma_paragram'] + \
                test_df['pr_lstma_wiki']) / 3

pred_test_y = (pr_mean_test>bt).astype(int)
out_df = pd.DataFrame({"qid":test_df["qid"].values})
out_df['prediction'] = pred_test_y
out_df.to_csv("submission.csv", index=False)