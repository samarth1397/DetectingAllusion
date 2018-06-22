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
from treeFunctions import *
from dependencies import *

	

class detect:
	
	def __init__(self,inputFolder='../data',outputFolder='../output/',dependencies='/home/users2/mehrotsh/scripts/packages/',language='english',cores=10):
		self.potential=inputFolder+'potential/'
		self.new=inputFolder+'new/'
		self.pickled=outputFolder+'pickle/'
		self.booksList=os.listdir(self.potential)
		self.dependencies=dependencies+'stanford-corenlp-full-2018-02-27/'
		# self.books=dict()
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
		
	def splitChunks(self,text):
		n = self.cores
		num = float(len(text))/n
		l = [ text[i:i + int(num)] for i in range(0, (n-1)*int(num), int(num))]
		l.append(text[(n-1)*int(num):])
		return l	
	

			
	def filterWithJacard(self,textChunks,books,threshold=0.2):

		mapInput=[(textChunks[i],books,self.booksList) for i in range(len(textChunks))]
	
		pool=Pool(processes=self.cores)
		results=pool.map(calcJacardChunk,mapInput)
		
		jacardScores=[]
		for scoreChunk in results:
			for score in scoreChunk:
				jacardScores.append(score)
				
		#pickling_on = open(self.output+'pickled/jacardScores.pickle',"wb")
		#pickle.dump(jacardScores, pickling_on)
		
		for sent in jacardScores:
			for book in self.booksList:
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
  
		return reducedBooks
		
			
	def filterWithTFIDF(self):
		pass
	
	
	def parseNewBook(self,textChunks):
		# mapInput=[(textChunks[i],self.dependencies) for i in range(len(textChunks))]
		pool=Pool(processes=self.cores)
		results=pool.map(parseNewText,textChunks)
		parseTrees=list()
		parsedSentences=list()
		for i in range(len(results)):
			parseTrees.append(results[i][0])
			parsedSentences.append(results[i][1])
		return (parseTrees,parsedSentences)
	
	def parseCandidates(self,reducedBooks):
		booksToBeParsed=[reducedBooks[bk] for bk in self.booksList]
		pool=Pool(processes=self.cores)
		results=pool.map(parseCandidateBooks,booksToBeParsed)
		potentialParseTrees=dict()
		potentialParsedSentences=dict()
		i=0
		for bk in self.booksList:
			potentialParseTrees[bk]=results[i][0]
			potentialParsedSentences[bk]=results[i][1]
			i=i+1
		return (potentialParseTrees,potentialParsedSentences)


	def syntacticScoring(self,parseTrees,potentialParseTrees):
		mapInput=[(parseTrees[i],potentialParseTrees,self.booksList) for i in range(len(parseTrees))]
		pool=Pool(processes=self.cores)
		results=pool.map(scoreSyntax,mapInput)
		print(len(results))
		allScores=list()
		for scoreChunk in results:
			for score in scoreChunk:
				allScores.append(score)
		print('allScores: ',len(allScores))
		return allScores
			
	def semanticScoring(self,text,reducedBooks):
		semanticScore=list()
		for i in range(len(text)):
			scoreDict=dict()
			s1v=avg_feature_vector(text[i],model,300,index2word_set)
			for bk in self.booksList:
				df=list()
				for j in range(len(reducedBooks[bk])):
					s2v=avg_feature_vector(reducedBooks[bk][j],model,300,index2word_set)
					semScore=1 - spatial.distance.cosine(s1v, s2v)
					df.append(semScore)
				scoreDict[bk]=df
			semanticScore.append(scoreDict)
		return semanticScore

	def aggeregateScoring(self,syntacticScore,semanticScore):
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

	def finalFiltering(self,scoreTuples,reducedBooks):
		totalPotentialSentences=0
		for bk in self.booksList:
			totalPotentialSentences=totalPotentialSentences+len(reducedBooks[bk])
		scoreTuples.sort(key=lambda tup: tup[0])
		finalTuples=list()
		k=0
		i=0
		while i<len(scoreTuples):
			senttups=scoreTuples[i:i+totalPotentialSentences]
			senttups.sort(key=lambda tup: tup[5],reverse=True)
			if senttups[0][5]>0.89:
				finalTuples.append(senttups[0])
			i=i+totalPotentialSentences
			k=k+1
		return finalTuples

	def nounBasedRanking(self,finalTuples,text,reducedBooks):
		newTuples=list()
		for tup in finalTuples:
			originalSent=text[tup[0]]
			refSent=reducedBooks[tup[1]][tup[2]]
			nounScore=jacardNouns(originalSent,refSent)
			newTuples.append(tup+(nounScore,))
		newTuples.sort(key=itemgetter(6,5),reverse=True)
		return newTuples	

def main():
	d=detect(inputFolder='../data/temp/')
	text=d.loadNew()
	books=d.loadCandidates()
	textChunks=d.splitChunks(text)
	
	print('Filtering chunk')
	reducedBooks=d.filterWithJacard(textChunks,books,threshold=0.2)
	pickling_on = open('../output/'+'testPackage/reducedBooks.pickle',"wb")
	pickle.dump(reducedBooks, pickling_on)

	print('Text: ',len(text))
	print('original is',len(books['isaiah']))
	print('reduced isaiah',len(reducedBooks['isaiah']))

	print('textChunks: ',len(textChunks))


	print('Parsing')
	parseTrees,parsedSentences=d.parseNewBook(textChunks)
	pickling_on = open('../output/'+'testPackage/parseTrees.pickle',"wb")
	pickle.dump(parseTrees, pickling_on)

	print('Parse trees',len(parseTrees))

	potentialParseTrees,potentialParsedSentences=d.parseCandidates(reducedBooks)
	print(len(parseTrees))
	# print(len(parseTrees['isaiah']))
	pickling_on = open('../output/'+'testPackage/potentialParseTrees.pickle',"wb")
	pickle.dump(potentialParseTrees, pickling_on)

	print('Potential Parse Trees isaiah ',len(potentialParseTrees['isaiah']))

	print('Moschitti')
	syntacticScore=d.syntacticScoring(parseTrees,potentialParseTrees)
	pickling_on = open('../output/'+'testPackage/allScores.pickle',"wb")
	pickle.dump(syntacticScore, pickling_on)

	print('syntactic: ',len(syntacticScore))
	


	'''
	pickle_off = open("../output/testPackage/reducedBooks.pickle","rb")
	reducedBooks = pickle.load(pickle_off)

	pickle_off = open("../output/testPackage/allScores.pickle","rb")
	syntacticScore = pickle.load(pickle_off)
	
	'''

	semanticScore=d.semanticScoring(text,reducedBooks)

	print('Semantic Score: ',len(semanticScore))

	scoreTuples=d.aggeregateScoring(syntacticScore,semanticScore)

	print(len(scoreTuples))

	pickling_on = open('../output/'+'testPackage/scoreTuples.pickle',"wb")
	pickle.dump(scoreTuples, pickling_on)

	finalTuples=d.finalFiltering(scoreTuples,reducedBooks)
	orderedTuples=d.nounBasedRanking(finalTuples,text,reducedBooks)
	
	pickling_on = open('../output/'+'testPackage/orderedTuples.pickle',"wb")
	pickle.dump(orderedTuples, pickling_on)



if __name__=="__main__":
	main()
	
		
		
		
		
