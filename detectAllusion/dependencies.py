from stanfordcorenlp import StanfordCoreNLP
import gensim
import nltk
import string
from nltk.stem import WordNetLemmatizer

# global nlp
nlp=StanfordCoreNLP('/home/users2/mehrotsh/scripts/packages/stanford-corenlp-full-2018-02-27/')
model = gensim.models.KeyedVectors.load_word2vec_format('../../GoogleNews-vectors-negative300.bin', binary=True) 
index2word_set = set(model.wv.index2word)
tokenizer = nltk.tokenize.TreebankWordTokenizer()
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(string.punctuation)
lemmatizer=WordNetLemmatizer()



