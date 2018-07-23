from stanfordcorenlp import StanfordCoreNLP
import gensim
import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import spacy

# Load stanford nlp: Change file path
# nlp=StanfordCoreNLP('/home/users2/mehrotsh/scripts/packages/stanford-corenlp-full-2018-02-27/',memory='8g',lang='en',timeout=1000000000) 
nlp=StanfordCoreNLP('/home/users2/mehrotsh/scripts/packages/stanford-corenlp-full-2018-02-27/',memory='8g',lang='de',timeout=1000000000) 

# English word2vec: Change file path
# model = gensim.models.KeyedVectors.load_word2vec_format('../../GoogleNews-vectors-negative300.bin', binary=True) 
# index2word_set = set(model.wv.index2word)

# German word2vec
model = gensim.models.KeyedVectors.load_word2vec_format('/home/users2/mehrotsh/Downloads/german.model', binary=True) 
index2word_set = set(model.wv.index2word)

# tokenizer = nltk.tokenize.TreebankWordTokenizer()
tokenizer = RegexpTokenizer(r'\w+')


# Stopword English
# stopwords = nltk.corpus.stopwords.words('english')
stopwords = nltk.corpus.stopwords.words('german')

stopwords.extend(string.punctuation)

# Load spacy for lemmatization and POS Tagging in German: Load all language modules
sp_de=spacy.load('de',disable=['parser','ner','textcat','entity'])
sp_en=spacy.load('en', disable=['parser','ner','textcat','entity'])
# sp=spacy.load('de', disable=['parser','ner','textcat','entity'])


