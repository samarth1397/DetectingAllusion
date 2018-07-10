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


#Useful functions

def tree(): 
    return defaultdict(tree)


def _leadingSpaces_(target):
    return len(target) - len(target.lstrip())

def _findParent_(curIndent, parid, treeRef):
    tmpid = parid
    while (curIndent <= treeRef[tmpid]['indent']):
        tmpid = treeRef[tmpid]['parid']
    return tmpid


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
    output = nlp.annotate(rawSentence, properties={'annotators': 'tokenize,ssplit,pos,parse','outputFormat': 'json','timeout':'50000'})
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
    tokens_a = [lemmatizer.lemmatize(token.lower().strip(string.punctuation)) for token in tokenizer.tokenize(a) if token.lower().strip(string.punctuation) not in stopwords]
    tokens_b = [lemmatizer.lemmatize(token.lower().strip(string.punctuation)) for token in tokenizer.tokenize(b) if token.lower().strip(string.punctuation) not in stopwords]
    if len(set(tokens_a).union(tokens_b))==0:
        ratio=0
    else:
        ratio = len(set(tokens_a).intersection(tokens_b)) / float(len(set(tokens_a).union(tokens_b)))
    return ratio    


def calcJacardChunk(chunkTuples):
    # print('computing chunk')
    chunk=chunkTuples[0]
    books=chunkTuples[1]
    booksList=chunkTuples[2]
    scoresChunk=list()
    for para in chunk:
        scoresDict={}
        for book in booksList:
            bookScore=[]
            for k in range(len(books[book])):
                simScore=jacardScore(para, books[book][k])
                bookScore.append((simScore,k))
            scoresDict[book]=bookScore
        scoresChunk.append(scoresDict)
    return scoresChunk


def parseNewText(paraChunk):
    # print('Parsing chunk')
    parseChunk=list()
    parseWithoutTokenChunk=list()
    for para in paraChunk:
        paraParse=list()
        paraWithoutTokenParse=list()
        para=sent_tokenize(para)
        for sent in para:
            sentParse=getNLPToks(sent)
            tempTree=tree()
            tempTree2=tree()
            generateTree(sentParse['parse'],tempTree)
            generateTree(sentParse['parse'],tempTree2)
#             parseSentenceChunk.append(sentParse['parse'])
            flipTree(tempTree)
            flipTree(tempTree2)
            paraParse.append(tempTree)
            paraWithoutTokenParse.append(removeTokens(tempTree2,sent))
        parseChunk.append(paraParse)
        parseWithoutTokenChunk.append(paraWithoutTokenParse)
    return parseChunk,parseWithoutTokenChunk   


    
def parseCandidateBooks(candidate):
    pTrees=list()
    pWithoutTokenTrees=list()
    for para in candidate:
        para=sent_tokenize(para)
        sentTrees=list()
        sentWithoutTokenTrees=list()
        for sent in para:
            sentParse=getNLPToks(sent)
            tempTree=tree()
            tempTree2=tree()
            generateTree(sentParse['parse'],tempTree)
            generateTree(sentParse['parse'],tempTree2)
            flipTree(tempTree)
            flipTree(tempTree2)
            sentTrees.append(tempTree)
            sentWithoutTokenTrees.append(removeTokens(tempTree2,sent))
        pTrees.append(sentTrees)
        pWithoutTokenTrees.append(sentWithoutTokenTrees)
    return pTrees,pWithoutTokenTrees


def scoreSyntax(chunkTuple):
    trChunks=chunkTuple[0]
    potentialParseTrees=chunkTuple[1]
    booksList=chunkTuple[2]    
    chunkDicts=list()
    for paraTrees in trChunks:
        sentScoreDict=dict()
        for book in booksList:
            bookTrees=potentialParseTrees[book]
            df=list()
            for bParaTree in bookTrees:
                s=0
                i=0
                for tr in paraTrees:
                    for bTree in bParaTree:
                        (rscore_st, nscore_st) = MoschittiPT(tr, bTree, 0.8, 1, 1)
                        s=s+nscore_st
                        i=i+1
                df.append(s/i)
            sentScoreDict[book]=df
        chunkDicts.append(sentScoreDict)
    return chunkDicts
def avg_feature_vector(sentence, model, num_features, index2word_set):
    words=tokenizer.tokenize(sentence)
    words=[lemmatizer.lemmatize(word.lower()) for word in words]
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

def avg_feature_vector_without_stopwords(sentence, model, num_features, index2word_set):
    words=tokenizer.tokenize(sentence)
    # words = sentence.split()
    words = [lemmatizer.lemmatize(token.lower().strip(string.punctuation)) for token in words if token.lower().strip(string.punctuation) not in stopwords]
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec  

def avg_feature_vector_nouns(sentence, model, num_features, index2word_set):
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
    for word in nouns:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec  

def avg_feature_vector_verbs(sentence, model, num_features, index2word_set):
    words=tokenizer.tokenize(sentence)
    words=[lemmatizer.lemmatize(word.lower()) for word in words]
    # words = sentence.split()
    # words = [token.lower().strip(string.punctuation) for token in tokenizer.tokenize(sentence) if token.lower().strip(string.punctuation) not in stopwords]
    verbs=[]
    for word,pos in nltk.pos_tag(words):
        if pos.startswith('VB'):
            verbs.append(word.lower().strip(string.punctuation))   

    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in verbs:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec  



def jacardNouns(sent1,sent2):
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
#     print(nouns1)
#     print(nouns2)
    if len(set(nouns1).union(nouns2))==0:
        ratio=0
    else:
        ratio = len(set(nouns1).intersection(nouns2)) / float(len(set(nouns1).union(nouns2)))        
    return ratio

def jacardVerbs(sent1,sent2):
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
#     print(nouns1)
#     print(nouns2)
    if len(set(nouns1).union(nouns2))==0:
        ratio=0
    else:
        ratio = len(set(nouns1).intersection(nouns2)) / float(len(set(nouns1).union(nouns2)))        
    return ratio

def jacardAdj(sent1,sent2):
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
#     print(nouns1)
#     print(nouns2)
    if len(set(nouns1).union(nouns2))==0:
        ratio=0
    else:
        ratio = len(set(nouns1).intersection(nouns2)) / float(len(set(nouns1).union(nouns2)))        
    return ratio



def longestSubsequence(a, b):
    a=tokenizer.tokenize(a)
    b=tokenizer.tokenize(b)
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


def commonProperNouns(sent1,sent2):
    sent1_tokens=nltk.pos_tag(tokenizer.tokenize(sent1))
    sent2_tokens=nltk.pos_tag(tokenizer.tokenize(sent2))
    sent1_proper=[word.lower() for (word,tag) in sent1_tokens if tag=='NNP']
    sent2_proper=[word.lower() for (word,tag) in sent2_tokens if tag=='NNP']
    common=len(set(sent1_proper).intersection(sent2_proper))
    return common