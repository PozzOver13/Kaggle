########################################################
# PROGRAM:  03_QUORA_pre_processing.py
# DATE:     2018-11-28
# NOTE:     
########################################################

#### LINK INTERESSANTI ####
# https://datascience.blog.wzb.eu/2016/07/13/autocorrecting-misspelled-words-in-python-using-hunspell/

#### INIZIALIZZAZIONE ####
import numpy as np
import pandas as pd
import seaborn as sns
from autocorrect import spell
from gensim.models import KeyedVectors
from tqdm import tqdm
tqdm.pandas()
import operator 
import re

color = sns.color_palette()
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

#### CARICAMENTO DATI ####
train_df = pd.read_csv("C:/Users/cg08900/Documents/Pandora/Personale/kaggle/Quora/datainput/train.csv")
test_df = pd.read_csv("C:/Users/cg08900/Documents/Pandora/Personale/kaggle/Quora/datainput/test.csv")
print("Train shape : ", train_df.shape)
print("Test shape : ", test_df.shape)

# lettura diversi embeddings
def load_embed(file):
    def get_coefs(word,*arr): 
        return word, np.asarray(arr, dtype='float32')
    
    if file == 'datainput/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec':
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file) if len(o)>100)
    else:
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))
        
    return embeddings_index

embd_file = 'datainput/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
embeddings_index = load_embed(embd_file)
'''
Embeddings disponibili
embd_wiki = 'datainput/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
embd_glove = 'datainput/embeddings/glove.840B.300d/glove.840B.300d.txt'
embd_google = 'datainput/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
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
def build_vocab(sentences, verbose =  True):
    """
    :param sentences: list of list of words
    :return: dictionary of words and their count
    """
    vocab = {}
    for sentence in tqdm(sentences, disable = (not verbose)):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

# pulisco le frasi dalla punteggiatura
# table = str.maketrans({key: None for key in string.punctuation})
# train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: x.translate(table))

def split_pro_str(x):
    x = re.findall(r"[\w']+|[.,!?;]", x)
    return(x)

sentences = train_df["question_text"].progress_apply(lambda x: split_pro_str(x))
vocab = build_vocab(sentences)
print({k: vocab[k] for k in list(vocab)[:11]})

# controllo la coperturasull'embedding
def check_coverage(vocab, embeddings_index):
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in tqdm(vocab):
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except:

            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x

oov = check_coverage(vocab,embeddings_index)

# rimuovo punteggiatura e simboli
def clean_text(x):

    x = str(x)
    for punct in "/-'":
        x = x.replace(punct, ' ')
    for punct in '&':
        x = x.replace(punct, f' {punct} ')
    for punct in '?!.,"#$%\()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        x = x.replace(punct, '')
    return x



train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: clean_text(x))
sentences = train_df["question_text"].apply(lambda x: x.split())
vocab = build_vocab(sentences)

oov = check_coverage(vocab,embeddings_index)

# ripulisco i numeri maggiori di 9
def clean_numbers(x):

    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x

train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: clean_numbers(x))
sentences = train_df["question_text"].progress_apply(lambda x: x.split())
vocab = build_vocab(sentences)

oov = check_coverage(vocab,embeddings_index)

# creo una guida per alcune parole che vengono scritte in piu' modi in inglese
def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re


mispell_dict = {'colour':'color',
                'centre':'center',
                'didnt':'did not',
                'doesnt':'does not',
                'isnt':'is not',
                'shouldnt':'should not',
                'favourite':'favorite',
                'travelling':'traveling',
                'counselling':'counseling',
                'theatre':'theater',
                'cancelled':'canceled',
                'labour':'labor',
                'organisation':'organization',
                'wwii':'world war 2',
                'citicise':'criticize',
                'instagram': 'social medium',
                'whatsapp': 'social medium',
                'snapchat': 'social medium'

                }
mispellings, mispellings_re = _get_mispell(mispell_dict)

def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)

train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: replace_typical_misspell(x))
sentences = train_df["question_text"].progress_apply(lambda x: x.split())
to_remove = ['a','to','of','and']
sentences = [[word for word in sentence if not word in to_remove] for sentence in tqdm(sentences)]
vocab = build_vocab(sentences)

oov = check_coverage(vocab,embeddings_index)

# ripulisco ancora i misspelled
def correct_misspelled(x):
    xs = x.split()
    xs_s = [spell(item) for item in xs]
    q_out = ' '.join(xs_s)
    
    return q_out

train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: correct_misspelled(x))
sentences = train_df["question_text"].progress_apply(lambda x: x.split())
vocab = build_vocab(sentences)

type(sentences[1])

test_in = sentences[1]
test_in[2] = 'hvae'


test_l = []
for word in test_in:
    print(word)
    #test_l += spell(word) 




df_prova = train_df[0:2]

miss_q = df_prova['question_text'][0].split()
miss_q[3] = 'natoinalists'
miss_q_c = ' '.join(miss_q)
df_prova['question_text'][0] = miss_q_c


df_prova_spell = df_prova

df_prova_spell["question_text"] = df_prova_spell["question_text"].progress_apply(lambda x: correct_misspelled(x))


q1 = df_prova_spell['question_text'][0]

correct_misspelled(q1)
xs = q1.split()
[spell(item) for item in xs]

xs 

vocab_misspelled = correct_misspelled(sentences)

for sentence in sentences:
    print(sentence)








print(spell('mastrubation'))

prova = train_df['question_text'][0:1]

prova_split = prova[0].split(' ')

prova_split[0] = 'colrO'

print(map(spell, prova_split))

[spell(item) for item in prova_split]













