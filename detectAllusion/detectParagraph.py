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
from treeFunctionsParagraph import *
from dependencies import *


class detectParagraph:

	def __init__(self, inputFolder='../data/',outputFolder='../output/',dependencies='/home/users2/mehrotsh/scripts/packages/',language='english',cores=40):
		self.potential=inputFolder+'potential/'
		self.new=inputFolder+'new/'
		self.pickled=outputFolder+'pickle/'
		self.booksList=os.listdir(self.potential)
		self.dependencies=dependencies+'stanford-corenlp-full-2018-02-27/'
		self.cores=cores

	
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
		books=dict()
		for file in self.booksList:
			print(file)
			candidate=open(self.potential+file)
			rawtext=candidate.read()
			rawtext = strip_headers(rawtext).strip()
			candidate=rawtext.replace('\n',' ')
			candidate=rawtext.replace(':','. ')
			candidate=sent_tokenize(candidate)
			candidate = list(filter(lambda x: len(x)>5, candidate))
			books[file]=candidate
		return books		


	def splitNewPara(self,text,numOfSents=3):
		i=0
		textPara=list()
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
		return textPara


	def splitCandidatesPara(self,books,numOfSents=3):
		booksPara=dict()
		for file in self.booksList:
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
		return booksPara

	def splitChunks(self,textPara):
		n = self.cores
		num = float(len(textPara))/n
		l = [ textPara[i:i + int(num)] for i in range(0, (n-1)*int(num), int(num))]
		l.append(textPara[(n-1)*int(num):])
		return l	

	def splitChunks(self,text):
		n = self.cores
		num = float(len(text))/n
		l = [ text[i:i + int(num)] for i in range(0, (n-1)*int(num), int(num))]
		l.append(text[(n-1)*int(num):])
		return l	
	

			
	def filterWithJacard(self,textChunks,booksPara,threshold=0.30):

		mapInput=[(textChunks[i],booksPara,self.booksList) for i in range(len(textChunks))]
	
		pool=Pool(processes=self.cores)
		results=pool.map(calcJacardChunk,mapInput)
		
		jacardScores=[]
		for scoreChunk in results:
			for score in scoreChunk:
				jacardScores.append(score)
				
		#pickling_on = open(self.output+'pickled/jacardScores.pickle',"wb")
		#pickle.dump(jacardScores, pickling_on)
		
		for para in jacardScores:
			for book in self.booksList:
				para[book]=list(filter(lambda tup: tup[0]>threshold,para[book]))
		
		reducedPara=dict()
		for book in self.booksList:
			reducedPara[book]=list()		
		
		for para in jacardScores:
			for book in self.booksList:
				reducedPara[book]=reducedPara[book]+[x[1] for x in para[book]]
		
		for book in self.booksList:
			reducedPara[book]=list(set(reducedPara[book]))
		
		reducedParagraphs=dict()
		for book in self.booksList:
			reducedParagraphs[book]=list()
	
		for book in self.booksList:
			for para in reducedPara[book]:
				reducedParagraphs[book].append(booksPara[book][para])
  
		return reducedParagraphs
		

	def parseNewBook(self,textChunks):
		# mapInput=[(textChunks[i],self.dependencies) for i in range(len(textChunks))]
		pool=Pool(processes=self.cores)
		results=pool.map(parseNewText,textChunks)
		parseTrees=list()
		# parsedSentences=list()
		for i in range(len(results)):
			parseTrees.append(results[i])
		return parseTrees

	def parseCandidates(self,reducedParagraphs):
		booksToBeParsed=[reducedParagraphs[bk] for bk in self.booksList]
		pool=Pool(processes=len(reducedParagraphs))
		results=pool.map(parseCandidateBooks,booksToBeParsed)
		potentialParseTrees=dict()
		i=0
		for bk in self.booksList:
			potentialParseTrees[bk]=results[i]
			i=i+1
		return potentialParseTrees

	def syntacticScoring(self,parseTrees,potentialParseTrees):
		mapInput=[(parseTrees[i],potentialParseTrees,self.booksList) for i in range(len(parseTrees))]
		pool=Pool(processes=self.cores)
		results=pool.map(scoreSyntax,mapInput)
		print(len(results))
		syntacticScore=list()
		for scoreChunk in results:
			for score in scoreChunk:
				syntacticScore.append(score)
		print('syntacticScore: ',len(syntacticScore))
		return syntacticScore

	def semanticScoring(self,textPara,reducedParagraphs):
		semanticScore=list()
		for i in range(len(textPara)):
			scoreDict=dict()
			s1v=avg_feature_vector(textPara[i],model,300,index2word_set)
			for bk in self.booksList:
				df=list()
				for j in range(len(reducedParagraphs[bk])):
					s2v=avg_feature_vector(reducedParagraphs[bk][j],model,300,index2word_set)
					semScore=1 - spatial.distance.cosine(s1v, s2v)
					df.append(semScore)
				scoreDict[bk]=df
			semanticScore.append(scoreDict)
		return semanticScore


	def aggregateScoring(self,syntacticScore,semanticScore):
		scoreTuples=list()
		for i in range(len(syntacticScore)):
			synScore=syntacticScore[i]
			simScore=semanticScore[i]
			avgScore=dict()
			for bk in self.booksList:
				synScore_book=synScore[bk]
				simScore_book=simScore[bk]
				for k in range(len(synScore_book)):
					sy=synScore_book[k]
					sm=simScore_book[k]
					scoreTuples.append((i,bk,k,sy,sm,(sy+sm)/2))
		return scoreTuples

	def finalFiltering(self,scoreTuples,reducedParagraphs,threshold=0.75):
		totalPotentialSentences=0
		for bk in self.booksList:
			totalPotentialSentences=totalPotentialSentences+len(reducedParagraphs[bk])
		scoreTuples.sort(key=lambda tup: tup[0])
		finalTuples=list()
		k=0
		i=0
		while i<len(scoreTuples):
			senttups=scoreTuples[i:i+totalPotentialSentences]
			senttups.sort(key=lambda tup: tup[5],reverse=True)
			if senttups[0][5]>threshold:
				finalTuples.append(senttups[0])
			i=i+totalPotentialSentences
			k=k+1
		return finalTuples

	def nounBasedRanking(self,finalTuples,textPara,reducedParagraphs):
		newTuples=list()
		for tup in finalTuples:
			originalPara=textPara[tup[0]]
			refPara=reducedParagraphs[tup[1]][tup[2]]
			nounScore=jacardNouns(originalPara,refPara)
			newTuples.append(tup+(nounScore,))
		newTuples.sort(key=itemgetter(6,5),reverse=True)
		return newTuples	