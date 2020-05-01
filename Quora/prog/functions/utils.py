# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 06:52:56 2019

@author: CG08900
"""

# >> functions

def clean_numbers(sentence):
    return(re.sub(r"\d{2,}(\.[0-9]+)*", 
                  lambda x: len(x.group())*"#", 
                  sentence))

def freq(df, col):
    s_group_sizes = df\
        .groupby(col)\
        .agg('size')
    total = df.shape[0]
    df_out = pd.DataFrame({'n_rec': s_group_sizes,
                           '%': s_group_sizes / total})
    return df_out

def read_seq_embedding(file, max_lines = None):
    out_index = {}
    line_counter = 0
    for line in open(file):
        line_counter = line_counter +1
        out_index[line.split(" ")[0]] = np.asarray(line.split(" ")[1:], 
                                                   dtype='float32')
        if max_lines is not None:
            if line_counter >= max_lines: break
    print("Read {0} lines".format(line_counter))
    return out_index

def create_word_count(corpus, to_lowercase = False, add_to_filters = None):
    tokenFilter = TOKEN_FILTERS
    if add_to_filters is not None:
        tokenFilter = tokenFilter + add_to_filters
    tokenizer = Tokenizer(lower=to_lowercase, filters=tokenFilter)
    tokenizer.fit_on_texts(corpus)
    return(tokenizer.word_counts)
    
def clean_corpus(corpus, to_lower=True):
    corpus = [clean_numbers(sentence) for sentence in corpus]
    if (to_lower):
        corpus = [sentence.lower() for sentence in corpus]
    return(corpus)
    
def read_and_filter_embedding(file, word_index, source):
    
    vocabularySet = set(word_index.keys())
    
    if source.lower() == 'glove':
        # read sequential embeddings       
        filteredEmbedding = {}
        lineCounter = 0
        for line in open(file):
            lineCounter += 1
            word = line.split(" ")[0]
            if word in vocabularySet:
                filteredEmbedding[word] = np.asarray(line.split(" ")[1:], 
                                                     dtype='float32')   
                
    elif source.lower() in ['paragram', 'wiki']:
        # read sequential embeddings       
        filteredEmbedding = {}
        lineCounter = 0
        for line in open(file, encoding='latin'):
            lineCounter += 1
            word = line.split(" ")[0]
            if word in vocabularySet:
                filteredEmbedding[word] = np.asarray(line.split(" ")[1:], 
                                                     dtype='float32')   
    elif source.lower() == 'google':
        # read google word2vec with gensim
        embedding = gs.models.KeyedVectors\
            .load_word2vec_format(file, binary=True)
        embeddingSet = set(embedding.vocab)
        filteredEmbedding = {}
        for name in embeddingSet.intersection(vocabularySet):
            filteredEmbedding[name] = embedding[name]
            
    return(filteredEmbedding)
    
def check_coverage(tokenizer_or_wordcount, embedding, rows_to_print = 10):
    
    # create word count vocabulary
    if str(type(tokenizer_or_wordcount)) == "<class 'keras_preprocessing.text.Tokenizer'>":
        wordCounts = tokenizer_or_wordcount.word_counts
    elif str(type(tokenizer_or_wordcount)) in ["<class 'dict'>",
            "<class 'collections.OrderedDict'>"]:
        wordCounts = tokenizer_or_wordcount
    else:
        raise ValueError('Wrong class for tokenizer_or_wordcount')
    
    # create "out of vocabulary" dictionary
    oov = {}
    
    for name in set(wordCounts).difference(set(embedding)):
        oov[name]= wordCounts[name]
    
    # final statistics
    oovPercentSingleWord = len(oov)/len(wordCounts)*100
    oovPercentAll = sum(oov.values())/sum(wordCounts.values())*100
    
    oovToPrint = pd.DataFrame(sorted(oov.items(), key=lambda kv: kv[1], 
                                     reverse=True)[0:(rows_to_print-1)],
                              columns=['Word', 'N'])
    
    # print results
    print("Percentage of words not in embedding: {0} ({2} % of vocabulary)\n\n.{1}"\
          .format(oovPercentAll, oovToPrint, oovPercentSingleWord))
    

def fit_tokenizer(corpus, embedding_dict, max_features, 
                  maxlen, embed_size=300, add_filters=None):
    
    # set tokenizer filters
    filtersForTokenizer = TOKEN_FILTERS
    if add_filters is not None:
        filtersForTokenizer=filtersForTokenizer+add_filters
        
    # tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features, filters=filtersForTokenizer)
    tokenizer.fit_on_texts(corpus)
    tokenizedCorpus = pad_sequences(tokenizer.texts_to_sequences(corpus), 
                                    maxlen=maxlen)
    
    # prepare embedding matrix
    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(0, 0.1, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embedding_dict.get(word)
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector
        
    return({'tokenizer': tokenizer, 
            'train_matrix': tokenizedCorpus, 
            'embedding_matrix': embedding_matrix})

def apply_tokenizer(tokenizer, corpus, maxlen):
    tokenizedCorpus = pad_sequences(tokenizer.texts_to_sequences(corpus), 
                                    maxlen=maxlen)
    return(tokenizedCorpus)
    
    
def correct_spelling(x, dic):
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x


def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    
    for p in punct:
        text = text.replace(p, f' {p} ')
    
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  
    for s in specials:
        text = text.replace(s, specials[s])
    
    return text



def clean_misspelled_and_special(corpus_in):
    dict_misspelled_words = {'colour': 'color', 
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
                'Pokémon':'Pokemon',
                'pokémon':'pokemon'}
    
    corp_cl = [correct_spelling(sentence,
                                dict_misspelled_words) for sentence in corpus_in]
    
    
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", 
                     "€": "e", "™": "tm", "√": " sqrt ", 
                 "×": "x", "²": "2", "—": "-", "–": "-", 
                 "’": "'", "_": "-", "`": "'", '“': '"', 
                 '”': '"', '“': '"', "£": "e", '∞': 'infinity', 
                 'θ': 'theta', '÷': '/', 
                 'α': 'alpha', '•': '.', 'à': 'a', 
                 '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi'}
    
    corp_cl_out = [clean_special_chars(sentence, 
                                       punct, 
                                       punct_mapping) for sentence in corp_cl]
    
    
    return(corp_cl_out)


   