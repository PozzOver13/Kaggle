########################################################
# PROGRAM:  03_QUORA_pre_processing.py
# DATE:     2018-11-28
# NOTE:     
########################################################

#### LINK INTERESSANTI ####
# https://datascience.blog.wzb.eu/2016/07/13/autocorrecting-misspelled-words-in-python-using-hunspell/

#### INIZIALIZZAZIONE ####
import os
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.models import KeyedVectors
from tqdm import tqdm
tqdm.pandas()
import operator 
import re
import unidecode

color = sns.color_palette()
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

#### CARICAMENTO DATI ####
os.chdir('C:/Users/cg08900/Documents/Pandora/Personale/kaggle/Quora/')
train_df = pd.read_csv("datainput/train.csv")
test_df = pd.read_csv("datainput/test.csv")
print("Train shape : ", train_df.shape)
print("Test shape : ", test_df.shape)

# caricamento embedding
#news_path = 'C:/Users/cg08900/Documents/Pandora/Personale/kaggle/Quora/datainput/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
#embeddings_index = KeyedVectors.load_word2vec_format(news_path, binary=False)

# lettura diversi embeddings
def load_embed(file):
    def get_coefs(word,*arr): 
        return word, np.asarray(arr, dtype='float32')
    
    if file == 'datainput/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec':
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file) if len(o)>100)
    else:
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))
        
    return embeddings_index

embd_file = 'datainput/embeddings/glove.840B.300d/glove.840B.300d.txt'
embeddings_index = load_embed(embd_file)

'''
Embeddings disponibili
embd_wiki = 'datainput/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
embd_glove = 'datainput/embeddings/glove.840B.300d/glove.840B.300d.txt'
embd_google = 'datainput/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
embd_paragram = 'datainput/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
'''

## Comprendere l'embedding e la sua struttura
#  essendo il mapping parola -> numpy 1D array e' esattamente come un 
#  dizionario

prova = {}
prova['one'] = embeddings_index['one']
prova["°C"] = embeddings_index["°C"]
prova[","] = embeddings_index[","]
prova


#### PREPROCESSING KAGGLE ####
def build_vocab(sentence):
    
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

def check_coverage(vocab, embeddings_index):
    known_words = {}
    unknown_words = {}
    nb_known_words = 0
    nb_unknown_words = 0
    for word in vocab.keys():
        try:
            known_words[word] = embeddings_index[word]
            nb_known_words += vocab[word]
        except:
            unknown_words[word] = vocab[word]
            nb_unknown_words += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(known_words) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))
    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]

    return unknown_words

mispell_dict = {'colour': 'color', 
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
                'Pokémon':'Pokemon'}

def correct_spelling(x, dic):
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", 
                 "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', 
                 '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 
                 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi'}

def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    
    for p in punct:
        text = text.replace(p, f' {p} ')
    
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])
    
    return text

def split_pro_str(x):
    x = re.findall(r"[\w]+|[.,'!?;]", x)
    return(x)


train_df['question_text_cl'] = train_df['question_text'].progress_apply(lambda x: correct_spelling(x, mispell_dict))
train_df['question_text_cl'] = train_df['question_text_cl'].progress_apply(lambda x: clean_special_chars(x, punct, punct_mapping))
train_df['question_text_cl'] = train_df['question_text_cl'].progress_apply(lambda x: unidecode.unidecode(x)) # potenziale errore con altre lingue

sentences = train_df["question_text_cl"].progress_apply(lambda x: split_pro_str(x))
vocabx = build_vocab(sentences)
print({k: vocabx[k] for k in list(vocabx)[:11]})

oov = check_coverage(vocabx, embeddings_index)
oov[0:100]


controllo = train_df[:200]
build_vocab(senteces[190])
