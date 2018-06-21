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


def parseNewText(chunk):
    #print('Parsing chunk')
    parseChunk=list()
    parseSentenceChunk=list()
    for sent in chunk:
        sentParse=getNLPToks(sent)
        tempTree=tree()
        generateTree(sentParse['parse'],tempTree)
        parseSentenceChunk.append(sentParse['parse'])
        flipTree(tempTree)
        parseChunk.append(tempTree)
    return (parseChunk,parseSentenceChunk)		



class detect:
	
	def __init__(self,inputFolder='../data/',outputFolder='../output/',dependencies='/home/users2/mehrotsh/scripts/packages/',language='english',cores=40):
		self.potential=inputFolder+'potential/'
		self.new=inputFolder+'new/'
		self.pickled=outputFolder+'pickle/'
		self.booksList=os.listdir(self.potential)
		self.stopwords = nltk.corpus.stopwords.words('english')
		self.stopwords.extend(string.punctuation)
		self.nlp=StanfordCoreNLP(dependencies+'stanford-corenlp-full-2018-02-27/')
		self.tokenizer = nltk.tokenize.TreebankWordTokenizer()
		self.books=dict()
		self.cores=40
	
	def loadNew(self):
		test=os.listdir(self.new)[0]
		testB=open(self.new+test)
		raw=testB.read()
		text = strip_headers(raw).strip()
		text=text.replace('\n',' ')
		text=text.replace(':','. ')
		text=sent_tokenize(text)
		text = list(filter(lambda x: len(x)>5, text))	
		return text
		
	def loadCandidates(self):
		#books=dict()
		for file in self.booksList:
			print(file)
			candidate=open(self.potential+file)
			rawtext=candidate.read()
			rawtext = strip_headers(rawtext).strip()
			candidate=rawtext.replace('\n',' ')
			candidate=rawtext.replace(':','. ')
			candidate=sent_tokenize(candidate)
			candidate = list(filter(lambda x: len(x)>5, candidate))
			self.books[file]=candidate
		return self.books		
		
	def splitChunks(self,text):
		textChunks=[]
		j=0
		for i in range(self.cores+1):
			if (j+math.floor(len(text)/self.cores))<len(text):
				textChunks.append(text[j:j+math.floor(len(text)/self.cores)])
				j=j+math.floor(len(text)/self.cores)
			else:
				textChunks.append(text[j:len(text)])
		return textChunks
			
	
	def calcJacardChunk(self,chunk):
		#print('computing chunk')
		scoresChunk=list()
		for sent in chunk:
		    scoresDict={}
		    for book in self.booksList:
		        bookScore=[]
		        for k in range(len(self.books[book])):
		            simScore=jacardScore(sent, self.books[book][k])
		            bookScore.append((simScore,k))
		        scoresDict[book]=bookScore
		    scoresChunk.append(scoresDict)
		return scoresChunk
			
			
	def filterWithJacard(self,textChunks,threshold=0.2):
		print('Filtering out sentences using Jacard coefficient of word overlap')
		
		pool=Pool(processes=self.cores+1)
		results=pool.map(self.calcJacardChunk,textChunks)
		
		jacardScores=[]
		for scoreChunk in results:
			for score in scoreChunk:
				jacardScores.append(score)
				
		pickling_on = open(self.output+'pickled/jacardScores.pickle',"wb")
		pickle.dump(jacardScores, pickling_on)
		
		for sent in jacardScores:
			for book in booksList:
				sent[book]=list(filter(lambda tup: tup[0]>threshold,sent[book]))
		
		reducedSentences=dict()
		for book in self.booksList:
    		reducedSentences[book]=list()		
		
		for sent in jacardScores:
    		for book in self.booksList:
        		reducedSentences[book]=reducedSentences[book]+[x[1] for x in sent[book]]
		
		for book in self.booksList:
    		reducedSentences[book]=list(set(reducedSentences[book]))
		
		reducedBooks=dict()
		for book in self.booksList:
    		reducedBooks[book]=list()
	
		for book in self.booksList:
    		for sent in reducedSentences[book]:
        		reducedBooks[book].append(books[book][sent])
      
        pickling_on = open(self.output+'pickled/reducedBooks.pickle',"wb")
		pickle.dump(reducedBooks, pickling_on)
		return reducedBooks
		
			
	def filterWithTFIDF():
		pass
	
	
	def parseNewBook(self,text):
		pool=Pool(processes=self.cores+1)
		results=pool.map(parseNewText,textChunks)
		for i in range(len(results)):
    		parseTrees.append(results[i][0])
    		parsedSentences.append(results[i][1])
	
	def parseCandidates():
		pass
		
		
	def syntacticScoring():
		pass
		
	def semanticScoring():
		pass
		
	def aggeregateScoring():
		pass
		
	def finalFiltering():
		pass


def main():
	d=detect()
	text=d.loadNew()
	books=d.loadCandidates()
	textChunks=d.splitChunks(text)
	reducedBooks=filterWithJacard(textChunks,threshold=0.5)
	
	

	
if __name__=="__main__":
	main()
	
		
		
		
		
