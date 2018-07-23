import numpy as np
from nltk.corpus import wordnet as wn
from stanfordcorenlp import StanfordCoreNLP
import re
import bisect
from collections import defaultdict
import ast
import os
from gutenberg.cleanup import strip_headers
from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
import math
import gensim
import pickle
from scipy import spatial
from nltk.tree import *
import nltk.corpus
import nltk.tokenize.punkt
import nltk.stem.snowball
import string
from multiprocessing import Pool
from nltk.draw.tree import TreeView
from fuzzywuzzy import fuzz
from multiprocessing import Pool
from nltk import word_tokenize,pos_tag
from nltk.corpus import wordnet 
from operator import itemgetter
from dependencies import *
from itertools import islice
import io

#Useful functions

'''
Parse trees will stored as default dictionaries
'''

def tree(): 
    return defaultdict(tree)


def _leadingSpaces_(target):
    return len(target) - len(target.lstrip())

def _findParent_(curIndent, parid, treeRef):
    tmpid = parid
    while (curIndent <= treeRef[tmpid]['indent']):
        tmpid = treeRef[tmpid]['parid']
    return tmpid



'''
A function to convert Stanford NLP tokens and tags into default dictionaries
'''

def generateTree(rawTokens, treeRef):

    # (token
    REGEX_OPEN = r"^\s*\(([a-zA-Z0-9_']*)\s*$"
    # (token (tok1 tok2) (tok3 tok4) .... (tokx toky))
    REGEX_COMP = r"^\s*\(([a-zA-Z0-9_']+)\s*((?:[(]([a-zA-Z0-9_;.,?'!]+)\s*([a-zA-Z0-9_;\.,?!']+)[)]\s*)+)"    
    # (, ,) as stand-alone. Used for match() not search()
    REGEX_PUNC = r"^\s*\([,!?.'\"]\s*[,!?.'\"]\)"
    # (tok1 tok2) as stand-alone
    REGEX_SOLO_PAIR = r"^\s*\(([a-zA-Z0-9_']+)\s*([a-zA-Z0-9_']+)\)"
    # (tok1 tok2) used in search()
    REGEX_ISOL_IN_COMP = r"\(([a-zA-Z0-9_;.,?!']+)\s*([a-zA-Z0-9_;.,?!']+)\)"
    # (punc punc) used in search()
    REGEX_PUNC_SOLO = r"\([,!?.'\"]\s*[,!?.'\"]\)"
   
    treeRef[len(treeRef)] = {'curid':0, 
                             'parid':-1, 
                             'posOrTok':'ROOT', 
                             'indent':0,
                            'children':[],
                            'childrenTok':[]}
    ID_CTR = 1
    
    for tok in rawTokens[1:]:
        
        curIndent = _leadingSpaces_(tok) 
        parid = _findParent_(curIndent, ID_CTR-1, treeRef)
        
        # CHECK FOR COMPOSITE TOKENS
        checkChild = re.match(REGEX_COMP, tok)
        if (checkChild):
            treeRef[ID_CTR] = {'curid':ID_CTR, 
                               'parid':parid, 
                               'posOrTok':checkChild.group(1), 
                               'indent':curIndent,
                              'children':[],
                              'childrenTok':[]}
            upCTR = ID_CTR
            ID_CTR += 1
            
            subCheck = re.sub(REGEX_PUNC_SOLO,'',checkChild.group(2))
            subs = re.findall(REGEX_ISOL_IN_COMP, subCheck) 
            for ch in subs:
                treeRef[ID_CTR] = {'curid':ID_CTR, 
                                   'parid':upCTR, 
                                   'posOrTok':ch[0], 
                                   'indent':curIndent+2,
                                  'children':[],
                                  'childrenTok':[]}
                ID_CTR += 1
                treeRef[ID_CTR] = {'curid':ID_CTR, 
                                   'parid':ID_CTR-1, 
                                   'posOrTok':ch[1], 
                                   'indent':curIndent+2,
                                  'children':[],
                                  'childrenTok':[]}
                ID_CTR += 1
            continue
           

            
        checkSingle = re.match(REGEX_SOLO_PAIR, tok)
        if (checkSingle):
            treeRef[ID_CTR] = {'curid':ID_CTR, 
                               'parid':parid, 
                               'posOrTok':checkSingle.group(1), 
                               'indent':curIndent+2,
                              'children':[],
                              'childrenTok':[]}
            ID_CTR += 1
            treeRef[ID_CTR] = {'curid':ID_CTR, 
                               'parid':ID_CTR-1, 
                               'posOrTok':checkSingle.group(2), 
                               'indent':curIndent+2,
                              'children':[],
                              'childrenTok':[]}
            ID_CTR += 1
            continue
        
        
        checkPunc = re.match(REGEX_PUNC, tok)   
        if (checkPunc): # ignore punctuation
            continue

        checkMatch = re.match(REGEX_OPEN, tok)
        if (checkMatch):
            treeRef[ID_CTR] = {'curid':ID_CTR, 
                               'parid':parid, 
                               'posOrTok':checkMatch.group(1), 
                               'indent':curIndent,
                              'children':[],
                              'childrenTok':[]}
            ID_CTR += 1
            continue

    return
            

def flipTree(treeRef):
    # Pass 1 fill in children
    for k,v in treeRef.items():
        if (k > 0):
            bisect.insort(treeRef[v['parid']]['children'], k)
    # Pass 2 map children to tokens
    for k,v in treeRef.items():
        if (k > 0):
            treeRef[k]['childrenTok'] = [treeRef[ch]['posOrTok'] for ch in treeRef[k]['children']]
    treeRef[0]['childrenTok'] = treeRef[1]['posOrTok']
    
    
    
def _isLeaf_(tree, parentNode):
    return (len(tree[parentNode]['children']) == 0)

def _isPreterminal_(tree, parentNode):
    for idx in tree[parentNode]['children']:
        if not _isLeaf_(tree, idx):
            return False
    return True


'''
Implementation of the Partial Tree (PT) Kernel from:
"Efficient Convolution Kernels for Dependency and Constituent Syntactic Trees"
by Alessandro Moschitti
'''


def _deltaP_(tree1, tree2, seq1, seq2, store, lam, mu, p):

#     # Enumerate subsequences of length p+1 for each child set
    if p == 0:
        return 0
    else:
        # generate delta(a,b)
        _delta_(tree1, tree2, seq1[-1], seq2[-1], store, lam, mu)
        if store[seq1[-1], seq2[-1]] == 0:
            return 0
        else:
            runningTot = 0
            for i in range(p-1, len(seq1)-1):
                for r in range(p-1, len(seq2)-1):
                    scaleFactor = pow(lam, len(seq1[:-1])-i+len(seq2[:-1])-r)
                    dp = _deltaP_(tree1, tree2, seq1[:i], seq2[:r], store, lam, mu, p-1)
                    runningTot += (scaleFactor * dp)
            return runningTot

def _delta_(tree1, tree2, node1, node2, store, lam, mu):

    # No duplicate computations
    if store[node1, node2] >= 0:
        return

    # Leaves yield similarity score by definition
    if (_isLeaf_(tree1, node1) or _isLeaf_(tree2, node2)):
        store[node1, node2] = 0
        return

    # same parent node
    if tree1[node1]['posOrTok'] == tree2[node2]['posOrTok']:

        if _isPreterminal_(tree1, node1) and _isPreterminal_(tree2, node2):
            if tree1[node1]['childrenTok'] == tree2[node2]['childrenTok']:
                store[node1, node2] = lam
            else:
                store[node1, node2] = 0
            return

        else:
            # establishes p_max
            childmin = min(len(tree1[node1]['children']), len(tree2[node2]['children']))
            deltaTot = 0
            for p in range(1,childmin+1):
                # compute delta_p
                deltaTot += _deltaP_(tree1, tree2,
                                     tree1[node1]['children'],
                                     tree2[node2]['children'], store, lam, mu, p)

            store[node1, node2] = mu * (pow(lam,2) + deltaTot)
            return

    else:
        # parent nodes are different
        store[node1, node2] = 0
        return

def _ptKernel_(tree1, tree2, lam, mu):
    # Fill the initial state of the store                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    store = np.empty((len(tree1), len(tree2)))
    store.fill(-1)

    # O(N^2) to compute the tree dot product
    for i in range(len(tree1)):
        for j in range(len(tree2)):
            _delta_(tree1, tree2, i, j, store, lam, mu)

    return store.sum()

'''
Returns a tuple w/ format: (raw, normalized)
If NORMALIZE_FLAG set to False, tuple[1] = -1
'''
def MoschittiPT(tree1, tree2, lam, mu, NORMALIZE_FLAG):
    raw_score = _ptKernel_(tree1, tree2, lam, mu)
    if (NORMALIZE_FLAG):
        t1_score = _ptKernel_(tree1, tree1, lam, mu)
        t2_score = _ptKernel_(tree2, tree2, lam, mu)
        return (raw_score,(raw_score / math.sqrt(t1_score * t2_score)))
    else:
        return (raw_score,-1)    
    

'''
Parse a sentence using stanford nlp
'''
    
def getNLPToks(rawSentence):
    output = nlp.annotate(rawSentence, properties={'annotators': 'tokenize,ssplit,pos,parse','outputFormat': 'json','timeout':'1000000000'})
    output=ast.literal_eval(output)
    tokens = output['sentences'][0]['tokens']
    parse = output['sentences'][0]['parse'].split("\n")
    return {
        'toks':tokens, 'parse':parse
    }
  
'''
Function which removes tokens from a parse tree
'''

def removeTokens(tr,sent):
    for key in tr.keys():
        parse=tr[key]
        childrenTok=parse['childrenTok']
        if type(childrenTok)==list:
            i=0
            for word in childrenTok:
                if word in sent.split():
                    childrenTok[i]='NULLWORD'
                i=i+1
        if type(childrenTok)==str:
            if childrenTok in sent.split():
                childrenTok='NULLWORD'
                i=i+1
        posOrTok=parse['posOrTok']
        if posOrTok in sent.split():
            parse['posOrTok']='NULLWORD'
    return tr

'''
Jaccard score between two sentences; used for initial filtering
'''
def jacardScore(a, b):
    # with lemmatization
    # tokens_a = [lemmatizer.lemmatize(token.lower().strip(string.punctuation)) for token in tokenizer.tokenize(a) if token.lower().strip(string.punctuation) not in stopwords]
    # tokens_b = [lemmatizer.lemmatize(token.lower().strip(string.punctuation)) for token in tokenizer.tokenize(b) if token.lower().strip(string.punctuation) not in stopwords]
    
    # without lemmatization
    # tokens_a=[token.lower().strip(string.punctuation) for token in tokenizer.tokenize(a) if token.lower().strip(string.punctuation) not in stopwords]
    # tokens_b=[token.lower().strip(string.punctuation) for token in tokenizer.tokenize(b) if token.lower().strip(string.punctuation) not in stopwords]

    # a=sp(a,disable=['parser','ner','textcat','entity'])
    # b=sp(b,disable=['parser','ner','textcat','entity'])
    tokens_a=[token.lemma_.lower() for token in a if ((token.lemma_.lower() not in stopwords) and (token.text.lower() not in stopwords))]
    tokens_b=[token.lemma_.lower() for token in b if ((token.lemma_.lower() not in stopwords) and (token.text.lower() not in stopwords))]

    if len(set(tokens_a).union(tokens_b))==0:
        ratio=0
    else:
        ratio = len(set(tokens_a).intersection(tokens_b)) / float(len(set(tokens_a).union(tokens_b)))
    return ratio    
'''
Calculates the jaccard score for each sentence in the chunk against all sentences in the potential texts. 
Returns: a list of dictionaries: [dict1,dict2,dict3....]

Each dict looks like this:
{	potential1:[0.1,0.0,0.3,0.9,....];
	potential2:[0.2,0.5,...............];xc
	.
	.
}

'''

def calcJacardChunk(chunkTuples):
    # print('computing chunk')
    chunk=chunkTuples[0]
    books=chunkTuples[1]
    booksList=chunkTuples[2]
    scoresChunk=list()
    for sent in chunk:
        scoresDict={}
        for book in booksList:
            bookScore=[]
            for k in range(len(books[book])):
                simScore=jacardScore(sent, books[book][k])
                bookScore.append((simScore,k))
            scoresDict[book]=bookScore
        scoresChunk.append(scoresDict)
    return scoresChunk

'''
A function to parse every sentence in this chunk. 
Returns 3 lists: 
parseChunk: a list of parse trees of sentences in the chunk
parseSentenceChunk: a list of parse trees in the stanford nlp format 
parseWithoutTokenChunk: a list of parse trees without the tokens for every sentence in the chunk
'''


def parseNewText(chunk):
    #print('Parsing chunk')
    # chunk=chunkTuple[0]
    # location=chunkTuple[1]
    # nlp=StanfordCoreNLP(location)
    parseChunk=list()
    parseSentenceChunk=list()
    parseWithoutTokenChunk=list()
    for sent in chunk:
        sentParse=getNLPToks(sent)
        tempTree=tree()
        tempTree2=tree()
        generateTree(sentParse['parse'],tempTree)
        generateTree(sentParse['parse'],tempTree2)
        parseSentenceChunk.append(sentParse['parse'])
        flipTree(tempTree)
        flipTree(tempTree2)
        parseChunk.append(tempTree)
        parseWithoutTokenChunk.append(removeTokens(tempTree2,sent))
    print('over')
    return (parseChunk,parseSentenceChunk,parseWithoutTokenChunk)      

'''
Parse every sentence in the candidate. 
Returns 3 lists:
pTrees: a list of parse trees of sentences in the candidate
pSents: a list of parse trees in the stanford nlp format 
pWithoutTokenTrees: a list of parse trees without the tokens for every sentence in the candidate
'''

def parseCandidateBooks(candidate):
    # print('parsing')
    pTrees=list()
    pSents=list()
    pWithoutTokenTrees=list()
    for sent in candidate:
        sentParse=getNLPToks(sent)
        tempTree=tree()
        tempTree2=tree()
        generateTree(sentParse['parse'],tempTree)
        generateTree(sentParse['parse'],tempTree2)
        pSents.append(sentParse['parse'])
        flipTree(tempTree)
        flipTree(tempTree2)
        pTrees.append(tempTree)
        pWithoutTokenTrees.append(removeTokens(tempTree2,sent))
    print('candidate')
    return (pTrees,pSents,pWithoutTokenTrees)

'''
Syntactic scoring between a chunk of parse trees and all the parse trees of the potential candidates. 
Returns a list of dictionaries: [dict1, dict2, dict3, ............]
Each dictionary is of the following format:
{
	potential1: [0.9,0.072,0.64,............]
	potential2: [0.7,0,9,0.4................]
}
,i.e. keys are names of potential books and the corresponding value is a list of syntactic similarity between the sentence from the chunk and all the sentences in the potential candidate.
'''

def scoreSyntax(chunkTuple):
    trChunks=chunkTuple[0]
    potentialParseTrees=chunkTuple[1]
    booksList=chunkTuple[2]
    chunkDicts=list()
    for tr in trChunks:
        sentScoreDict=dict()
        for book in booksList:
    #         print(file)
            bookTrees=potentialParseTrees[book]
            df=list()
            for bTree in bookTrees:
                try:
                    (rscore_st, nscore_st) = MoschittiPT(tr, bTree, 0.8, 1, 1)
                    df.append(nscore_st)
                except TypeError:
                    df.append(0)
    #         print(df)
            sentScoreDict[book]=df
        chunkDicts.append(sentScoreDict)
    print('scored')
    return chunkDicts

'''
Returns the average word vector of the sentence using the pretrained word2vec model
'''

def avg_feature_vector(sentence, model, num_features, index2word_set):
    # English
    # words=tokenizer.tokenize(sentence)
    # words=[lemmatizer.lemmatize(word.lower()) for word in words]
    
    # German
    # a=sp(sentence,disable=['parser','ner','textcat','entity'])
    words=[token.lemma_.lower() for token in sentence if token.pos_ != 'PUNCT']

    # words=[word.lower() for word in words]
    # words = sentence.split()
    # words = [token.lower().strip(string.punctuation) for token in tokenizer.tokenize(sentence) if token.lower().strip(string.punctuation) not in stopwords]
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec  

'''
Returns the average word vector of the sentence after the removal of stopwords using the pretrained word2vec model
'''

def avg_feature_vector_without_stopwords(sentence, model, num_features, index2word_set):
    # English
    # words=tokenizer.tokenize(sentence)
    # words = [lemmatizer.lemmatize(token.lower().strip(string.punctuation)) for token in words if token.lower().strip(string.punctuation) not in stopwords]
    # words = [token.lower().strip(string.punctuation) for token in words if token.lower().strip(string.punctuation) not in stopwords]

    # German
    # a=sp(sentence,disable=['parser','ner','textcat','entity'])
    words=[token.lemma_.lower() for token in sentence if ((token.lemma_.lower() not in stopwords) and (token.text.lower() not in stopwords))]

    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec  


'''
Returns the average word vector of the nouns in the sentence using the pretrained word2vec model
'''

def avg_feature_vector_nouns(sentence, model, num_features, index2word_set):
    
    # English
    # words=tokenizer.tokenize(sentence)
    # words=[lemmatizer.lemmatize(word.lower()) for word in words]
    # words=[word.lower() for word in words]
    # words = sentence.split()
    # words = [token.lower().strip(string.punctuation) for token in tokenizer.tokenize(sentence) if token.lower().strip(string.punctuation) not in stopwords]
    '''
    nouns=[]
    for word,pos in nltk.pos_tag(words):
        if pos.startswith('NN'):
            nouns.append(word.lower().strip(string.punctuation))   
    '''

    # German
    # a=sp(sentence,disable=['parser','ner','textcat','entity'])
    nouns=[token.lemma_.lower() for token in sentence if ((token.pos_ == 'NOUN') or (token.pos_ == 'PROPN')) ]

    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in nouns:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec  


'''
Returns the average word vector of the verbs in the sentence using the pretrained word2vec model
'''

def avg_feature_vector_verbs(sentence, model, num_features, index2word_set):
    
    # English
    # words=tokenizer.tokenize(sentence)
    # words=[lemmatizer.lemmatize(word.lower()) for word in words]
    # words=[word.lower() for word in words]
    # words = sentence.split()
    # words = [token.lower().strip(string.punctuation) for token in tokenizer.tokenize(sentence) if token.lower().strip(string.punctuation) not in stopwords]
    
    '''
    verbs=[]
    for word,pos in nltk.pos_tag(words):
        if pos.startswith('VB'):
            verbs.append(word.lower().strip(string.punctuation))   
    
    '''

    # German
    # a=sp(sentence,disable=['parser','ner','textcat','entity'])
    verbs=[token.lemma_.lower() for token in sentence if token.pos_ == 'VERB']

    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in verbs:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec  

'''
Returns the jaccard index of nouuns in the two sentences
'''

def jacardNouns(sent1,sent2):
    
    # English
    '''
    words1=tokenizer.tokenize(sent1)
    words2=tokenizer.tokenize(sent2)
    words_1=[lemmatizer.lemmatize(word.lower()) for word in words1]
    words_2=[lemmatizer.lemmatize(word.lower()) for word in words2]
    nouns1=[]
    for word,pos in nltk.pos_tag(words_1):
        if pos.startswith('NN'):
            nouns1.append(word.lower().strip(string.punctuation))
    nouns2=[]
    for word,pos in nltk.pos_tag(words_2):
        if pos.startswith('NN'):
            nouns2.append(word.lower().strip(string.punctuation))
    '''

    # German
    # a=sp(sent1,disable=['parser','ner','textcat','entity'])
    nouns1=[token.lemma_.lower() for token in sent1 if ((token.pos_ == 'NOUN') or (token.pos_ == 'PROPN'))]
    # b=sp(sent2)
    nouns2=[token.lemma_.lower() for token in sent2 if ((token.pos_ == 'NOUN') or (token.pos_ == 'PROPN'))]


    if len(set(nouns1).union(nouns2))==0:
        ratio=0
    else:
        ratio = len(set(nouns1).intersection(nouns2)) / float(len(set(nouns1).union(nouns2)))        
    return ratio


'''
Returns the jaccard index of verbs in the two sentences
'''

def jacardVerbs(sent1,sent2):

    # English
    '''
    words1=tokenizer.tokenize(sent1)
    words2=tokenizer.tokenize(sent2)
    words_1=[lemmatizer.lemmatize(word.lower()) for word in words1]
    words_2=[lemmatizer.lemmatize(word.lower()) for word in words2]
    nouns1=[]
    for word,pos in nltk.pos_tag(words_1):
        if pos.startswith('VB'):
            nouns1.append(word.lower().strip(string.punctuation))
    nouns2=[]
    for word,pos in nltk.pos_tag(words_2):
        if pos.startswith('VB'):
            nouns2.append(word.lower().strip(string.punctuation))
    '''
    
    # German
    # a=sp(sent1,disable=['parser','ner','textcat','entity'])
    # b=sp(sent2)
    nouns1=[token.lemma_.lower() for token in sent1 if token.pos_ == 'VERB']
    nouns2=[token.lemma_.lower() for token in sent2 if token.pos_ == 'VERB']

    if len(set(nouns1).union(nouns2))==0:
        ratio=0
    else:
        ratio = len(set(nouns1).intersection(nouns2)) / float(len(set(nouns1).union(nouns2)))        
    return ratio


'''
Returns the jaccard index of adjectives in the two sentences
'''

def jacardAdj(sent1,sent2):

    # English
    '''
    words1=tokenizer.tokenize(sent1)
    words2=tokenizer.tokenize(sent2)
    words_1=[lemmatizer.lemmatize(word.lower()) for word in words1]
    words_2=[lemmatizer.lemmatize(word.lower()) for word in words2]
    nouns1=[]
    for word,pos in nltk.pos_tag(words_1):
        if pos.startswith('JJ'):
            nouns1.append(word.lower().strip(string.punctuation))
    nouns2=[]
    for word,pos in nltk.pos_tag(words_2):
        if pos.startswith('JJ'):
            nouns2.append(word.lower().strip(string.punctuation))
    '''
    # a=sp(sent1,disable=['parser','ner','textcat','entity'])
    # b=sp(sent2,disable=['parser','ner','textcat','entity'])
    nouns1=[token.lemma_.lower() for token in sent1 if token.pos_ == 'ADJ']
    nouns2=[token.lemma_.lower() for token in sent2 if token.pos_ == 'ADJ']

    if len(set(nouns1).union(nouns2))==0:
        ratio=0
    else:
        ratio = len(set(nouns1).intersection(nouns2)) / float(len(set(nouns1).union(nouns2)))        
    return ratio


'''
Returns the longest subsequence of words between the two paragraphs
'''


def longestSubsequence(a, b):
    a=tokenizer.tokenize(a)
    b=tokenizer.tokenize(b)
    
    # Removing stopwords
    a=[w.lower() for w in a if w.lower() not in stopwords]
    b=[w.lower() for w in b if w.lower() not in stopwords]
    
    lengths = [[0 for j in range(len(b)+1)] for i in range(len(a)+1)]
    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i+1][j+1] = lengths[i][j] + 1
            else:
                lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])
    # read the substring out from the matrix
    result = ""
    x, y = len(a), len(b)
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x-1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y-1]:
            y -= 1
        else:
            assert a[x-1] == b[y-1]
            result = a[x-1] + " " +result
            x -= 1
            y -= 1
    return result

'''
Returns the number of common proper nouns between the two paragraphs
'''

def commonProperNouns(sent1,sent2):

    # English
    '''
    sent1_tokens=nltk.pos_tag(tokenizer.tokenize(sent1))
    sent2_tokens=nltk.pos_tag(tokenizer.tokenize(sent2))
    sent1_proper=[word.lower() for (word,tag) in sent1_tokens if tag=='NNP']
    sent2_proper=[word.lower() for (word,tag) in sent2_tokens if tag=='NNP']
    '''

    # German
    # a=sp(sent1,disable=['parser','ner','textcat','entity'])
    # b=sp(sent2,disable=['parser','ner','textcat','entity'])
    sent1_proper=[token.lemma_.lower() for token in sent1 if token.pos_ == 'PROPN']
    sent2_proper=[token.lemma_.lower() for token in sent2 if token.pos_ == 'PROPN']
    common=len(set(sent1_proper).intersection(sent2_proper))
    return common



'''
Loading fast text multilingual word vectors
'''

def load_vec(emb_path, nmax=50000):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id

'''
average fast text vector
'''
def fasttext_avg_feature_vector(sentence, embeddings, num_features, word2idset):
    words=tokenizer.tokenize(sentence)
    words=[lemmatizer.lemmatize(word.lower()) for word in words]
    # words = sentence.split()
    # words = [token.lower().strip(string.punctuation) for token in tokenizer.tokenize(sentence) if token.lower().strip(string.punctuation) not in stopwords]
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in word2idset:
            n_words += 1
            feature_vec = np.add(feature_vec, embeddings[word2idset[word]])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec  

'''
average fast text vector without stopwords
'''

def fasttext_avg_feature_vector_without_stopwords(sentence, embeddings, num_features, word2idset):
    words=tokenizer.tokenize(sentence)
    words = [lemmatizer.lemmatize(token.lower().strip(string.punctuation)) for token in words if lemmatizer.lemmatize(token.lower().strip(string.punctuation)) not in stopwords]
    # words = sentence.split()
    # words = [token.lower().strip(string.punctuation) for token in tokenizer.tokenize(sentence) if token.lower().strip(string.punctuation) not in stopwords]
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in word2idset:
            n_words += 1
            feature_vec = np.add(feature_vec, embeddings[word2idset[word]])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec  

'''
average fast text vector, nouns only; change pos-tagger to multilingual pos-tagger
'''
def fasttext_avg_feature_vector_nouns(sentence, embeddings, num_features, word2idset):
    words=tokenizer.tokenize(sentence)
    words=[lemmatizer.lemmatize(word.lower()) for word in words]
    # words = sentence.split()
    # words = [token.lower().strip(string.punctuation) for token in tokenizer.tokenize(sentence) if token.lower().strip(string.punctuation) not in stopwords]
    nouns=[]
    for word,pos in nltk.pos_tag(words):
        if pos.startswith('NN'):
            nouns.append(word.lower().strip(string.punctuation))  
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in word2idset:
            n_words += 1
            feature_vec = np.add(feature_vec, embeddings[word2idset[word]])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec  


'''
average fast text vector, verbs only; change pos-tagger to multilingual pos-tagger
'''
def fasttext_avg_feature_vector_verbs(sentence, embeddings, num_features, word2idset):
    words=tokenizer.tokenize(sentence)
    words=[lemmatizer.lemmatize(word.lower()) for word in words]
    # words = sentence.split()
    # words = [token.lower().strip(string.punctuation) for token in tokenizer.tokenize(sentence) if token.lower().strip(string.punctuation) not in stopwords]
    nouns=[]
    for word,pos in nltk.pos_tag(words):
        if pos.startswith('VB'):
            nouns.append(word.lower().strip(string.punctuation))  
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in word2idset:
            n_words += 1
            feature_vec = np.add(feature_vec, embeddings[word2idset[word]])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec  


