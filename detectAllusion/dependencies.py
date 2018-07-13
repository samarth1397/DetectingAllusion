from stanfordcorenlp import StanfordCoreNLP
import gensim
import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import spacy

# Load stanford nlp
nlp=StanfordCoreNLP('/home/users2/mehrotsh/scripts/packages/stanford-corenlp-full-2018-02-27/')

# English word2vec
model = gensim.models.KeyedVectors.load_word2vec_format('../../GoogleNews-vectors-negative300.bin', binary=True) 
index2word_set = set(model.wv.index2word)

# German word2vec
# model = gensim.models.KeyedVectors.load_word2vec_format('/home/users2/mehrotsh/Downloads/german.model', binary=True) 
# index2word_set = set(model.wv.index2word)

# tokenizer = nltk.tokenize.TreebankWordTokenizer()
tokenizer = RegexpTokenizer(r'\w+')


# Stopword English
stopwords = nltk.corpus.stopwords.words('english')

# Stopwords German
# stopwords = nltk.corpus.stopwords.words('german')

stopwords.extend(string.punctuation)

# English lemmatization
lemmatizer=WordNetLemmatizer()

# Load spacy for lemmatization and POS Tagging in German
# sp=spacy.load('de')
sp=spacy.load('en', disable=['parser','ner','textcat','entity'])


