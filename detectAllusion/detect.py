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
	
	def __init__(self,inputFolder='../data/',outputFolder='../output/',dependencies='/home/users2/mehrotsh/scripts/packages/',language='english',cores=40):
		self.potential=inputFolder+'potential/'
		self.new=inputFolder+'new/'
		self.pickled=outputFolder+'pickle/'
		self.booksList=os.listdir(self.potential)
		self.dependencies=dependencies+'stanford-corenlp-full-2018-02-27/'
		# self.books=dict()
		self.cores=cores
		self.output=outputFolder
	
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
	

			
	def filterWithJacard(self,textChunks,books,threshold=0.35):

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
  		
  		pool.close()
		return reducedBooks
		
			
	def filterWithTFIDF(self):
		pass
	
	
	def parseNewBook(self,textChunks):
		# mapInput=[(textChunks[i],self.dependencies) for i in range(len(textChunks))]
		pool=Pool(processes=self.cores)
		results=pool.map(parseNewText,textChunks)
		parseTrees=list()
		parsedSentences=list()
		parseWithoutTokenTrees=list()
		for i in range(len(results)):
			parseTrees.append(results[i][0])
			parsedSentences.append(results[i][1])
			parseWithoutTokenTrees.append(results[i][2])
		pool.close()
		return (parseTrees,parsedSentences,parseWithoutTokenTrees)
	
	def parseCandidates(self,reducedBooks):
		booksToBeParsed=[reducedBooks[bk] for bk in self.booksList]
		pool=Pool(processes=len(reducedBooks))
		results=pool.map(parseCandidateBooks,booksToBeParsed)
		potentialParseTrees=dict()
		potentialParsedSentences=dict()
		potentialParseWithoutTokenTrees=dict()
		i=0
		for bk in self.booksList:
			potentialParseTrees[bk]=results[i][0]
			potentialParsedSentences[bk]=results[i][1]
			potentialParseWithoutTokenTrees[bk]=results[i][2]
			i=i+1
		pool.close()
		return (potentialParseTrees,potentialParsedSentences,potentialParseWithoutTokenTrees)


	def syntacticScoring(self,parseTrees,potentialParseTrees,parseWithoutTokenTrees,potentialParseWithoutTokenTrees):
		mapInput=[(parseTrees[i],potentialParseTrees,self.booksList) for i in range(len(parseTrees))]
		pool=Pool(processes=self.cores)
		results=pool.map(scoreSyntax,mapInput)
		# print(len(results))
		syntaxScores=list()
		for scoreChunk in results:
			for score in scoreChunk:
				syntaxScores.append(score)
		# print('allScores: ',len(allScores))
		pool.close()
		print('Scoring with tokens complete. Scoring without tokens started.')
		mapInput=[(parseWithoutTokenTrees[i],potentialParseWithoutTokenTrees,self.booksList) for i in range(len(parseWithoutTokenTrees))]
		pool=Pool(processes=self.cores)
		results=pool.map(scoreSyntax,mapInput)
		# print(len(results))
		syntaxScoresWithoutTokens=list()
		for scoreChunk in results:
			for score in scoreChunk:
				syntaxScoresWithoutTokens.append(score)
		# print('allScores: ',len(allScores))
		pool.close()
		return syntaxScores,syntaxScoresWithoutTokens
			
	def semanticScoring(self,text,reducedBooks):
		semanticScore=list()
		for i in range(len(text)):
			scoreDict=dict()
			s1=avg_feature_vector(text[i],model,300,index2word_set)
			s1ws=avg_feature_vector_without_stopwords(text[i],model,300,index2word_set)
			s1n=avg_feature_vector_nouns(text[i],model,300,index2word_set)
			s1v=avg_feature_vector_verbs(text[i],model,300,index2word_set)
			for bk in self.booksList:
				df=list()
				for j in range(len(reducedBooks[bk])):
					s2=avg_feature_vector(reducedBooks[bk][j],model,300,index2word_set)
					s2ws=avg_feature_vector_without_stopwords(reducedBooks[bk][j],model,300,index2word_set)
					s2n=avg_feature_vector_nouns(reducedBooks[bk][j],model,300,index2word_set)
					s2v=avg_feature_vector_verbs(reducedBooks[bk][j],model,300,index2word_set)
					semScore=1 - spatial.distance.cosine(s1, s2)
					semScore_withoutStop=1 - spatial.distance.cosine(s1ws, s2ws)
					semScore_nouns=1 - spatial.distance.cosine(s1n, s2n)
					semScore_verbs=1 - spatial.distance.cosine(s1v, s2v)
					properNouns=commonProperNouns(text[i],reducedBooks[bk][j])
					df.append((semScore,semScore_withoutStop,semScore_nouns,semScore_verbs,properNouns))

				scoreDict[bk]=df
			semanticScore.append(scoreDict)
		return semanticScore

	def longestSubsequenceScoring(self,text,reducedBooks):
		lcs=list()
		lcsScore=list()
		for i in range(len(text)):
			scoreDict_lcs=dict()
			scoreDict_lcsScore=dict()
			for bk in self.booksList:
				df_lcs=list()
				df_lcsScore=list()
				for j in range(len(reducedBooks[bk])):
					data=[text[i],reducedBooks[bk][j]]
					subsequence=longestSubsequence(text[i],reducedBooks[bk][j])
					df_lcs.append(subsequence)
					df_lcsScore.append(len(subsequence.split()))
				scoreDict_lcs[bk]=df_lcs
				scoreDict_lcsScore[bk]=df_lcsScore
			lcs.append(scoreDict_lcs)
			lcsScore.append(scoreDict_lcsScore)
		return (lcsScore,lcs)

	def aggregateScoring(self,syntacticScore,semanticScore,lcsScore,lcsString,syntacticScoreWithoutTokens):
		scoreTuples=list()
		for i in range(len(syntacticScore)):
			synScore=syntacticScore[i]	#syntax
			simScore=semanticScore[i]	#semantic
			lcs_score=lcsScore[i]		#lcs
			lcs_string=lcsString[i]		# lcs string
			synWithoutTokenScore=syntacticScoreWithoutTokens[i]
			for bk in self.booksList:
				synScore_book=synScore[bk]
				simScore_book=simScore[bk]
				lcs_score_book=lcs_score[bk]
				lcs_string_book=lcs_string[bk]
				synWithoutTokenScore_book=synWithoutTokenScore[bk]
				for k in range(len(synScore_book)):
					sy=synScore_book[k]
					sm=simScore_book[k]
					lscore=lcs_score_book[k]
					lstring=lcs_string_book[k]
					syWithoutToken=synWithoutTokenScore_book[k]
					scoreTuples.append((i,bk,k,sy,sm[0],sm[1],sm[2],sm[3],(sy+sm[1])/2,lscore,lstring,syWithoutToken,sm[4]))
		return scoreTuples

	def finalFiltering(self,scoreTuples,reducedBooks,threshold=0.89):
		totalPotentialSentences=0
		for bk in self.booksList:
			totalPotentialSentences=totalPotentialSentences+len(reducedBooks[bk])
		scoreTuples.sort(key=lambda tup: tup[0])
		finalTuples=list()
		k=0
		i=0
		while i<len(scoreTuples):
			senttups=scoreTuples[i:i+totalPotentialSentences]
			senttups.sort(key=itemgetter(12,8),reverse=True)
			if senttups[0][8]>threshold:
				finalTuples.append(senttups[0])
			i=i+totalPotentialSentences
			k=k+1

		finalTuples.sort(key=lambda tup: tup[8])

		diffTuples=list()
		for tup in scoreTuples:
			if (tup[3]>0.8 and abs(tup[3]-tup[4])>=0.12) or (tup[4]>0.8 and abs(tup[3]-tup[4])>=0.12):
				diffTuples.append(tup)

		return finalTuples,diffTuples

	def nounBasedRanking(self,finalTuples,text,reducedBooks):
		newTuples=list()
		for tup in finalTuples:
			originalSent=text[tup[0]]
			refSent=reducedBooks[tup[1]][tup[2]]
			nounScore=jacardNouns(originalSent,refSent)
			verbScore=jacardVerbs(originalSent,refSent)
			adjScore=jacardAdj(originalSent,refSent)
			newTuples.append(tup+(nounScore,verbScore,adjScore))
		newTuples.sort(key=itemgetter(13,8),reverse=True)
		return newTuples


	def writeOutput(self,newTuples,text,reducedBooks):
		f=open(self.output+'nounSortedSentencePairs.txt','w')
		i=1
		lines=list()
		for t in newTuples:
			j=str(i)
			lines.append('Pairing: '+j)
			lines.append('\n')
			lines.append('New Sentence: '+text[t[0]])
			lines.append('\n')
			lines.append('Reference: \n'+reducedBooks[t[1]][t[2]])
			lines.append('\n')
			lines.append('Similar Sentence is from: '+str(t[1]))
			lines.append('\n')
			lines.append('Syntactic Score: '+str(t[3]))
			lines.append('\n')
			lines.append('Syntactic Similarity without tokens: '+str(t[11]))
			lines.append('\n')
			lines.append('Semantic Score: '+str(t[4]))
			lines.append('\n')
			lines.append('Semantic Score without stopwords: '+str(t[5]))
			lines.append('\n')
			lines.append('LCS Length: '+str(t[9]))
			lines.append('\n')
			lines.append('LCS: '+t[10])
			lines.append('\n')
			lines.append('Jaccard of common nouns: '+str(t[13]))
			lines.append('\n')
			lines.append('Jaccard of common verbs: '+str(t[14]))
			lines.append('\n')
			lines.append('Jaccard of common adjectives: '+str(t[15]))
			lines.append('\n')
			lines.append('Semantic similarity nouns: '+str(t[6]))
			lines.append('\n')
			lines.append('Semantic similarity verbs: '+str(t[7]))
			lines.append('\n\n')
			i=i+1
		f.writelines(lines)
		return
def main():
	d=detect(inputFolder='../data/temp/',outputFolder='../output/temp/')
	print('Loading books and splitting')
	text=d.loadNew()
	books=d.loadCandidates()
	textChunks=d.splitChunks(text)
	
	print('Filtering using Jaccard')
	reducedBooks=d.filterWithJacard(textChunks,books,threshold=0.4)
	pickling_on = open('../output/'+'temp/reducedBooks.pickle',"wb")
	pickle.dump(reducedBooks, pickling_on)

	# print('Text: ',len(text))
	# print('original is',len(books['isaiah']))
	# print('reduced isaiah',len(reducedBooks['isaiah']))

	# print('textChunks: ',len(textChunks))


	print('Syntactic parsing')
	parseTrees,parsedSentences,parseWithoutTokenTrees=d.parseNewBook(textChunks)
	pickling_on = open('../output/'+'temp/parseTrees.pickle',"wb")
	pickle.dump(parseTrees, pickling_on)

	# print('Parse trees',len(parseTrees))

	potentialParseTrees,potentialParsedSentences,potentialParseWithoutTokenTrees=d.parseCandidates(reducedBooks)
	# print(len(parseTrees))
	# print(len(parseTrees['isaiah']))
	pickling_on = open('../output/'+'temp/potentialParseTrees.pickle',"wb")
	pickle.dump(potentialParseTrees, pickling_on)

	# print('Potential Parse Trees isaiah ',len(potentialParseTrees['isaiah']))

	print('Moschitti scoring')
	syntacticScore,syntacticScoreWithoutTokens=d.syntacticScoring(parseTrees,potentialParseTrees,parseWithoutTokenTrees,potentialParseWithoutTokenTrees)
	pickling_on = open('../output/'+'temp/allScores.pickle',"wb")
	pickle.dump(syntacticScore, pickling_on)

	print('Semantic scoring')
	semanticScore=d.semanticScoring(text,reducedBooks)

	# print('Semantic Score: ',len(semanticScore))

	print('Extracting longest subsequence')
	lcsScore,lcs=d.longestSubsequenceScoring(text,reducedBooks)

	print('Average scoring')

	scoreTuples=d.aggregateScoring(syntacticScore,semanticScore,lcsScore,lcs,syntacticScoreWithoutTokens)

	# print(len(scoreTuples))


	pickling_on = open('../output/'+'temp/scoreTuples.pickle',"wb")
	pickle.dump(scoreTuples, pickling_on)

	finalTuples,diffTuples=d.finalFiltering(scoreTuples,reducedBooks,0.75)
	if len(finalTuples)>100:
		finalTuples=finalTuples[0:100]
	orderedTuples=d.nounBasedRanking(finalTuples,text,reducedBooks)
	
	pickling_on = open('../output/'+'temp/orderedTuples.pickle',"wb")
	pickle.dump(orderedTuples, pickling_on)

	print('Final results: \n\n\n')

	i=1
	for t in orderedTuples:
		print('Pairing: ',i)
		print('\n')
		print('New Sentence: ',text[t[0]])
		print('\n')
		print('Reference: \n',reducedBooks[t[1]][t[2]])
		print('\n')
		print('Similar Sentence is from: ',t[1])
		print('Syntactic Score: ',t[3])
		print('Syntactic Similarity without tokens: ',t[11])
		print('Semantic Score: ',t[4])
		print('Semantic Score without stopwords: ',t[5])
		print('LCS Length: ',t[9])
		print('LCS: ',t[10])
		print('Jaccard of common nouns: ',t[12])
		print('Jaccard of common verbs: ',t[13])
		print('Jaccard of common adjectives: ',t[14])
		print('Semantic similarity nouns: ',t[6])
		print('Semantic similarity verbs: ',t[7])
		print('\n\n')
		i=i+1

	# d.writeOutput(orderedTuples,text,reducedBooks)

	print('\n\n Tuples with large difference in syntactic and semantic value: \n\n\n')

	diffTuples=d.nounBasedRanking(diffTuples,text,reducedBooks)
	d.writeOutput(diffTuples,text,reducedBooks)
	pickling_on = open('../output/'+'temp/diffTuples.pickle',"wb")
	pickle.dump(diffTuples, pickling_on)
'''
	i=1
	for t in diffTuples:
		print('Pairing: ',i)
		print('\n')
		print('New Sentence: ',text[t[0]])
		print('\n')
		print('Reference: \n',reducedBooks[t[1]][t[2]])
		print('\n')
		print('Similar Sentence is from: ',t[1])
		print('Syntactic Score: ',t[3])
		print('Syntactic Similarity without tokens: ',t[11])
		print('Semantic Score: ',t[4])
		print('Semantic Score without stopwords: ',t[5])
		print('LCS Length: ',t[9])
		print('LCS: ',t[10])
		print('Jaccard of common nouns: ',t[12])
		print('Jaccard of common verbs: ',t[13])
		print('Jaccard of common adjectives: ',t[14])
		print('Semantic similarity nouns: ',t[6])
		print('Semantic similarity verbs: ',t[7])
		print('\n\n')
		i=i+1

'''


if __name__=="__main__":
	main()
	
		
		
		
		
