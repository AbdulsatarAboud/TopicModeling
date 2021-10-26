import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem.porter import *
import numpy as np

np.random.seed(2018)
import nltk


def lemmatize_stemming(text, pos):
    englishStemmer = SnowballStemmer("english")
    return englishStemmer.stem(WordNetLemmatizer().lemmatize(text, pos=get_wordnet_pos(pos)))


def preprocess(text):
    result = []
    tokens = gensim.utils.simple_preprocess(text)
    for token in nltk.pos_tag(tokens):
        if token[0] not in gensim.parsing.preprocessing.STOPWORDS and len(token[0]) > 3:
            result.append(lemmatize_stemming(token[0], token[1]))
            #result.append(token[1])
    return result


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
