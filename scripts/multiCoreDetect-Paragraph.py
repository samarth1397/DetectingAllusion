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
from collections import Counter    

public='/home/users2/mehrotsh/scripts/packages/stanford-corenlp-full-2018-02-27/'
personal='/home/samarth/stanford-corenlp-full-2018-02-27/'

nlp = StanfordCoreNLP(public)


#################################################### Functions ############################################################################

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
Implementation of the Colins-Duffy or Subset-Tree (SST) Kernel
'''

def _cdHelper_(tree1, tree2, node1, node2, store, lam, SST_ON):
    # No duplicate computations
    if store[node1, node2] >= 0:
        return

    # Leaves yield similarity score by definition
    if (_isLeaf_(tree1, node1) or _isLeaf_(tree2, node2)):
        store[node1, node2] = 0
        return

    # same parent node
    if tree1[node1]['posOrTok'] == tree2[node2]['posOrTok']:
        # same children tokens
        if tree1[node1]['childrenTok'] == tree2[node2]['childrenTok']:
            # Check if both nodes are pre-terminal
            if _isPreterminal_(tree1, node1) and _isPreterminal_(tree2, node2):
                store[node1, node2] = lam
                return
            # Not pre-terminal. Recurse among the children of both token trees.
            else:
                nChildren = len(tree1[node1]['children'])

                runningTotal = None
                for idx in range(nChildren):
                     # index ->  node_id
                    tmp_n1 = tree1[node1]['children'][idx]
                    tmp_n2 = tree2[node2]['children'][idx]
                    # Recursively run helper
                    _cdHelper_(tree1, tree2, tmp_n1, tmp_n2, store, lam, SST_ON)
                    # Set the initial value for the layer. Else multiplicative product.
                    if (runningTotal == None):
                        runningTotal = SST_ON + store[tmp_n1, tmp_n2]
                    else:
                        runningTotal *= (SST_ON + store[tmp_n1, tmp_n2])

                store[node1, node2] = lam * runningTotal
                return
        else:
            store[node1, node2] = 0
    else: # parent nodes are different
        store[node1, node2] = 0
        return


def _cdKernel_(tree1, tree2, lam, SST_ON):
    # Fill the initial state of the store
    store = np.empty((len(tree1), len(tree2)))
    store.fill(-1)
    # O(N^2) to compute the tree dot product
    for i in range(len(tree1)):
        for j in range(len(tree2)):
            _cdHelper_(tree1, tree2, i, j, store, lam, SST_ON)

    return store.sum()

'''
Returns a tuple w/ format: (raw, normalized)
If NORMALIZE_FLAG set to False, tuple[1] = -1
'''
def CollinsDuffy(tree1, tree2, lam, NORMALIZE_FLAG, SST_ON):
    raw_score = _cdKernel_(tree1, tree2, lam, SST_ON)
    if (NORMALIZE_FLAG):
        t1_score = _cdKernel_(tree1, tree1, lam, SST_ON)
        t2_score = _cdKernel_(tree2, tree2, lam, SST_ON)
        return (raw_score,(raw_score / math.sqrt(t1_score * t2_score)))
    else:
        return (raw_score,-1)



'''
Implementation of the Partial Tree (PT) Kernel from:
"Efficient Convolution Kernels for Dependency and Constituent Syntactic Trees"
by Alessandro Moschitti
'''

'''
The delta function is stolen from the Collins-Duffy kernel
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


def getNLPToks(rawSentence):
    output = nlp.annotate(rawSentence, properties={'annotators': 'tokenize,ssplit,pos,parse','outputFormat': 'json','timeout':'50000'})
    output=ast.literal_eval(output)
    tokens = output['sentences'][0]['tokens']
    parse = output['sentences'][0]['parse'].split("\n")
    return {
        'toks':tokens, 'parse':parse
    }
def avg_feature_vector(sentence, model, num_features, index2word_set):
#     words = sentence.split()
    words = [token.lower().strip(string.punctuation) for token in tokenizer.tokenize(sentence) if token.lower().strip(string.punctuation) not in stopwords]
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec
    
    
def getDuffyScore(sent1,sent2):
    tree_1=tree()
    tree_2=tree()
    out1=getNLPToks(sent1)
    out2=getNLPToks(sent2)
    generateTree(out1['parse'],tree_1)
    generateTree(out2['parse'],tree_2)
    flipTree(tree_1)
    flipTree(tree_2)
    (rscore_st, nscore_st) = CollinsDuffy(tree_1, tree_2, 0.8, 1, 1)
    return rscore_st,nscore_st    
    
    
def getMoschittiScore(sent1,sent2):
    tree_1=tree()
    tree_2=tree()
    out1=getNLPToks(sent1)
    out2=getNLPToks(sent2)
    generateTree(out1['parse'],tree_1)
    generateTree(out2['parse'],tree_2)
    flipTree(tree_1)
    flipTree(tree_2)
    (rscore_st, nscore_st) = MoschittiPT(tree_1, tree_2, 0.8, 1, 1)
#     return rscore_st,nscore_st
    return nscore_st  
    
def jacardScore(a, b):
    tokens_a = [token.lower().strip(string.punctuation) for token in tokenizer.tokenize(a) if token.lower().strip(string.punctuation) not in stopwords]
    tokens_b = [token.lower().strip(string.punctuation) for token in tokenizer.tokenize(b) if token.lower().strip(string.punctuation) not in stopwords]
    if len(set(tokens_a).union(tokens_b))==0:
        ratio=0
    else:
        ratio = len(set(tokens_a).intersection(tokens_b)) / float(len(set(tokens_a).union(tokens_b)))
    return ratio

def calcJacard(sent):
    scoresDict={}
    for book in booksList:
        bookScore=[]
        for k in range(len(books[book])):
            simScore=jacardScore(sent, books[book][k])
            bookScore.append((simScore,k))
        scoresDict[book]=bookScore
    return scoresDict    
    
def parseBook(candidate):
    pTrees=list()
#     pSents=list()
    for para in candidate:
        para=sent_tokenize(para)
        sentTrees=list()
        for sent in para:
            sentParse=getNLPToks(sent)
            tempTree=tree()
            generateTree(sentParse['parse'],tempTree)
    #         pSents.append(sentParse['parse'])
            flipTree(tempTree)
            sentTrees.append(tempTree)
        pTrees.append(sentTrees)
#     return (pTrees,pSents)
    return pTrees

def scoreSyntax(trChunks):
    chunkDicts=list()
    for paraTrees in trChunks:
        sentScoreDict=dict()
        for book in booksList:
            bookTrees=potentialParseTrees[book]
            df=list()
            for bSentTrees in bookTrees:
                s=0
                i=0
                for tr in paraTrees:
                    for bTree in bSentTrees:
                        (rscore_st, nscore_st) = MoschittiPT(tr, bTree, 0.8, 1, 1)
                        s=s+nscore_st
                        i=i+1
                df.append(s/i)
            sentScoreDict[book]=df
        chunkDicts.append(sentScoreDict)
    return chunkDicts

def calcJacardChunk(chunk):
    print('computing chunk')
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
    
def parseNewText(paraChunk):
    print('Parsing chunk')
    parseChunk=list()
    for para in paraChunk:
        paraParse=list()
        para=sent_tokenize(para)
        for sent in para:
            sentParse=getNLPToks(sent)
            tempTree=tree()
            generateTree(sentParse['parse'],tempTree)
#             parseSentenceChunk.append(sentParse['parse'])
            flipTree(tempTree)
            paraParse.append(tempTree)
        parseChunk.append(paraParse)
    return parseChunk   
    
############################################ Script ###############################################################################    


stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(string.punctuation)
stopwords.append('')
stopwords.append('thou')

tokenizer = nltk.tokenize.TreebankWordTokenizer()

##### Loading #########

potential="../data/potential/"
booksList=os.listdir(potential)

print('Loading books')

test="../data/new/newTestament.txt"
testB=open(test)
raw=testB.read()
text = strip_headers(raw).strip()
text=text.replace('\n',' ')
text=text.replace(':','. ')
text=sent_tokenize(text)
text = list(filter(lambda x: len(x)>5, text))
testB.close()

textPara=list()
numOfSents=3
i=0
while(i<len(text)):
    if((i+numOfSents)<len(text)):
        para=text[i:i+numOfSents]
        para=" ".join(para)
        i=i+1
        textPara.append(para)
    else:
        para=text[i:len(text)]
        para=" ".join(para)
        textPara.append(para)
        break
        
textChunks=list()        
cores=40
j=0
for i in range(cores+1):
    if (j+math.floor(len(textPara)/40))<len(textPara):
        textChunks.append(textPara[j:j+math.floor(len(textPara)/40)])
        j=j+math.floor(len(textPara)/40)
    else:
        textChunks.append(textPara[j:len(textPara)])        


books=dict()
for file in booksList:
    print(file)
    candidate=open(potential+file)
    rawtext=candidate.read()
    candidate.close()
    rawtext = strip_headers(rawtext).strip()
    candidate=rawtext.replace('\n',' ')
    candidate=rawtext.replace(':','. ')
    candidate=sent_tokenize(candidate)
    candidate = list(filter(lambda x: len(x)>5, candidate))
    books[file]=candidate


print('Loading completed')

booksPara=dict()
for file in booksList:
    candidate=books[file]
    i=0
    candidatePara=list()
    while(i<len(candidate)):
        if((i+numOfSents)<len(candidate)):
            para=candidate[i:i+numOfSents]
            para=" ".join(para)
            i=i+1
            candidatePara.append(para)
        else:
            para=candidate[i:len(candidate)]
            para=" ".join(para)
            candidatePara.append(para)
            break
    booksPara[file]=candidatePara


'''
print('Extending stop words')

combinedTexts=[]
for bk in booksList:
    combinedTexts.append(" ".join(books[bk]))
combinedTexts=" ".join(combinedTexts)  

tokens=nltk.word_tokenize(combinedTexts)
fdist=nltk.FreqDist(tokens)
contextualStopWords=[tup[0] for tup in fdist.most_common(75)]
stopwords=list(set(stopwords+contextualStopWords))

'''

####### Jacard #############

print('Computing Jacardian')


pool=Pool(processes=cores+1)
results=pool.map(calcJacardChunk,textChunks)

print(len(results))

jacardScores=[]

jacardScores=[]
for scoreChunk in results:
    for score in scoreChunk:
        jacardScores.append(score)

print('Pickling jacard')

pickling_on = open("../output/paragraphSimilarity/jacardScores.pickle","wb")
pickle.dump(jacardScores, pickling_on)

#pickle_off = open("./bible/jacardScores.pickle","rb")
#jacardScores = pickle.load(pickle_off)


print('Filtering using jacard')

for para in jacardScores:
    for book in booksList:
        para[book]=list(filter(lambda tup: tup[0]>0.2,para[book]))

reducedPara=dict()
for book in booksList:
    reducedPara[book]=list()
    
for para in jacardScores:
    for book in booksList:
        reducedPara[book]=reducedPara[book]+[x[1] for x in para[book]]

for book in booksList:
    reducedPara[book]=list(set(reducedPara[book]))

reducedParagraphs=dict()
for book in booksList:
    reducedParagraphs[book]=list()
    
for book in booksList:
    for para in reducedPara[book]:
        reducedParagraphs[book].append(booksPara[book][para])

pickling_on = open("../output/paragraphSimilarity/reducedBooks.pickle","wb")
pickle.dump(reducedBooks, pickling_on)


print('Reduced Isaiah',len(reducedBooks['isaiah']))

####### Parsing New book###############


print('Parsing new book')


pool=Pool(processes=cores+1)
results=pool.map(parseNewText,textChunks)

parseTrees=list()
parsedSentences=list()

for i in range(len(results)):
    parseTrees.append(results[i])
 
print(len(parseTrees))

pickling_on = open("../output/paragraphSimilairty/parseTrees.pickle","wb")
pickle.dump(parseTrees, pickling_on)


####### Parsing candidates #########

print('Parsing candidates')
potentialParseTrees=dict()


booksToBeParsed=[reducedBooks[bk] for bk in booksList]
pool=Pool(processes=len(reducedParagraphs))
results=pool.map(parseBook,booksToBeParsed)
print(len(results))


i=0
for bk in booksList:
    potentialParseTrees[bk]=results[i]
#     potentialParsedSentences[bk]=results[i][1]
    i=i+1

pickling_on = open("../output/paragraphSimilarity/potentialParseTrees.pickle","wb")
pickle.dump(potentialParseTrees, pickling_on)


'''

print('pickling')


pickle_off = open("../output/bible-temp-2/jacardScores.pickle","rb")
jacardScores = pickle.load(pickle_off)

pickle_off = open("../output/bible-temp-2/reducedBooks.pickle","rb")
reducedBooks = pickle.load(pickle_off)

pickle_off = open("../output/bible-temp-2/parseTrees.pickle","rb")
parseTrees = pickle.load(pickle_off)

pickle_off = open("../output/bible-temp-2/potentialParseTrees.pickle","rb")
potentialParseTrees = pickle.load(pickle_off)


print(len(potentialParseTrees['isaiah']))

'''
########## Syntactic Scoring - Moschitti #########

print('Syntactic Scoring')

pool=Pool(processes=cores+1)
results=pool.map(scoreSyntax,parseTrees)

allScores=list()
for scoreChunk in results:
    for score in scoreChunk:
        allScores.append(score)

pickling_on = open("../output/paragraphSimilarity/allScores.pickle","wb")
pickle.dump(allScores, pickling_on)


print('Preparing for semantic scoring')

model = gensim.models.KeyedVectors.load_word2vec_format('../../GoogleNews-vectors-negative300.bin', binary=True) 
index2word_set = set(model.wv.index2word)


print('Aggeregate scoring and sorting')

scoreTuples=list()
for i in range(len(allScores)):
    s1v=avg_feature_vector(text[i],model,300,index2word_set)
    for fl in booksList:
        scores=allScores[i][fl]
        for j in range(len(scores)):
            s2v=avg_feature_vector(reducedBooks[fl][j],model,300,index2word_set)
            semanticScore=1 - spatial.distance.cosine(s1v, s2v)
            scoreTuples.append((i,fl,j,scores[j],semanticScore,(scores[j]+semanticScore)/2))

scoreTuples.sort(key=lambda tup: tup[5],reverse=True)

print('Writing to a file')

pickling_on = open("../output/paragraphSimilarity/scoreTuples.pickle","wb")
pickle.dump(scoreTuples, pickling_on)

'''

f=open('./bible-temp-2/outputScores-2.txt','a')
for t in scoreTuples[0:500]:
    f.write('Original Sentence: '+text[t[0]])
    f.write('\n')
    f.write('Similar Sentence is from: '+t[1])
    f.write('\n')
#     file.write('Syntactic Score: '+t[3])
#     file.write('Semantic Score: '+t[4])
    f.write(reducedBooks[t[1]][t[2]])
    f.write('\n\n')


print('Quick removal of false positives')

sentNumbers=list()
for t in scoreTuples[0:500]:
    sentNumbers.append(t[0])

counts = Counter(sentNumbers)    
f=open('../output/bible-temp-2/outputScores-filtered.txt','a') 
    
for t in scoreTuples[0:500]:
    if counts[t[0]]>7:
#         print('skipped')
        continue
    else:
        f.write('Original Sentence: '+text[t[0]])
        f.write('\n')
        f.write('Similar Sentence is from: '+t[1])
        f.write('\n')
    #     file.write('Syntactic Score: '+t[3])
    #     file.write('Semantic Score: '+t[4])
        f.write(reducedBooks[t[1]][t[2]])
        f.write('\n\n')


'''































