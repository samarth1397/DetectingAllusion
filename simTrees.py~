import numpy as np
import math
from nltk.corpus import wordnet as wn
from stanfordcorenlp import StanfordCoreNLP
import re
import bisect
from collections import defaultdict
import ast
import os
from gutenberg.cleanup import strip_headers
from nltk.tokenize import sent_tokenize

public='/home/users2/mehrotsh/scripts/packages/stanford-corenlp-full-2018-02-27/'
personal='/home/samarth/stanford-corenlp-full-2018-02-27/'

nlp = StanfordCoreNLP(personal)

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
  
    # manually insert Root token 
    treeRef[len(treeRef)] = {'curid':0, 
                             'parid':-1, 
                             'posOrTok':'ROOT', 
                             'indent':0,
                            'children':[],
                            'childrenTok':[]}
    ID_CTR = 1
    
    for tok in rawTokens[1:]:
        
        curIndent = _leadingSpaces_(tok) # the current indent level
        parid = _findParent_(curIndent, ID_CTR-1, treeRef) # determine parid
        
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
    output = nlp.annotate(rawSentence, properties={'annotators': 'tokenize,ssplit,pos,parse','outputFormat': 'json'})
    output=ast.literal_eval(output)
    tokens = output['sentences'][0]['tokens']
    parse = output['sentences'][0]['parse'].split("\n")
    return {
        'toks':tokens, 'parse':parse
    }

test="./new/1.txt"
testB=open(test)
raw=testB.read()
text = strip_headers(raw).strip()
text=text.replace('\n',' ')
text=sent_tokenize(text)
text = list(filter(lambda x: len(x)>1, text))

parseTrees=list()
for sent in text:
    sentParse=getNLPToks(sent)
    tempTree=tree()
    generateTree(sentParse['parse'],tempTree)
    flipTree(tempTree)
    parseTrees.append(tempTree)
    
    
print('Parse trees created for new text\n\n')    

potential="./potential/"
potentialParseTrees=dict()

for fl in os.listdir(potential):
    print(fl)
    candidate=open(potential+fl)
    rawtext=candidate.read()
    rawtext = strip_headers(rawtext).strip()
    candidate=rawtext.replace('\n',' ')
    candidate=sent_tokenize(candidate)
    candidate=list(filter(lambda x: len(x)>1, candidate))
    pTrees=list()
    for sent in candidate:
        sentParse=getNLPToks(sent)
        tempTree=tree()
        generateTree(sentParse['parse'],tempTree)
        flipTree(tempTree)
        pTrees.append(tempTree)
    potentialParseTrees[fl]=pTrees
    
print('Parse trees created for potential candidates\n\n')    
    
    
allScores=list()
i=0
for tr in parseTrees:
    if i%10==0:
        print(i)
    sentScoreDict=dict()
    for fl in os.listdir(potential):
        bookTrees=potentialParseTrees[fl]
        df=list()
        for bTree in bookTrees:
            (rscore_st, nscore_st) = MoschittiPT(tr, bTree, 0.8, 1, 1)
            df.append(nscore_st)
        sentScoreDict[fl]=df
    allScores.append(sentScoreDict)
    i=i+1                                                  
          
print('Scoring completed\n\n')          
                
books=dict()
for fl in os.listdir(potential):
    print(fl)
    candidate=open(potential+fl)
    rawtext=candidate.read()
    rawtext = strip_headers(rawtext).strip()
    candidate=rawtext.replace('\n',' ')
    candidate=sent_tokenize(candidate)
    candidate=list(filter(lambda x: len(x)>1, candidate))
    books[fl]=candidate

for i in range(1,10):
    print('Sentence',i)
    print('Original Sent',text[i])
    for book in os.listdir(potential):
        print(book)
        maxIndex=allScores[i][book].index(max(allScores[i][book]))
        print('Score',allScores[i][book][maxIndex])
        print('Similar sentence:',books[book][maxIndex])
    print('\n\n')





