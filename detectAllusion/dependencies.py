from stanfordcorenlp import StanfordCoreNLP
import gensim
import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import spacy


'''
Stanford NLP 
- Change the file path 
- And comment/uncomment the appropriate model and index
'''

# nlp=StanfordCoreNLP('/home/users2/mehrotsh/scripts/packages/stanford-corenlp-full-2018-02-27/',memory='8g',lang='en',timeout=1000000000) 
nlp=StanfordCoreNLP('/home/users2/mehrotsh/scripts/packages/stanford-corenlp-full-2018-02-27/',memory='8g',lang='de',timeout=1000000000) 

'''
Word2vec: 
- Change the file path 
- And comment/uncomment the appropriate model and index
'''

# English word2vec: 
# model = gensim.models.KeyedVectors.load_word2vec_format('../../GoogleNews-vectors-negative300.bin', binary=True) 
# index2word_set = set(model.wv.index2word)

# German word2vec
model = gensim.models.KeyedVectors.load_word2vec_format('/home/users2/mehrotsh/Downloads/german.model', binary=True) 
index2word_set = set(model.wv.index2word)

# tokenizer = nltk.tokenize.TreebankWordTokenizer()
tokenizer = RegexpTokenizer(r'\w+')


'''
Stopwords: Change the file path and comment/uncomment the model that you want to load
'''

# stopwords = nltk.corpus.stopwords.words('english')
stopwords = nltk.corpus.stopwords.words('german')

stopwords.extend(string.punctuation)



'''
Spacy: load all the models; no need for commenting; specifiy the language while creating a detect/detectParagraph object
'''
sp_de=spacy.load('de',disable=['parser','ner','textcat','entity'])
sp_en=spacy.load('en', disable=['parser','ner','textcat','entity'])
# sp=spacy.load('de', disable=['parser','ner','textcat','entity'])


