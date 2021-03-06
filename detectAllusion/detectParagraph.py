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


'''
a class which can search for allusions between an input text and a set of candidate texts
'''

class detectParagraph:

	def __init__(self, inputFolder='../data/',outputFolder='../output/',dependencies='/home/users2/mehrotsh/scripts/packages/',language='en',cores=40):
		self.potential=inputFolder+'potential/'
		self.new=inputFolder+'new/'
		self.pickled=outputFolder+'pickle/'
		self.booksList=os.listdir(self.potential)
		self.dependencies=dependencies+'stanford-corenlp-full-2018-02-27/'
		self.cores=cores
		self.outputFolder=outputFolder
		self.language=language

	'''
	Loads the 'new' text and splits it into sentences
	Returns: [sent1,sent2,sent3,.........]
	'''
	
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

	'''
	Loads the candidate texts and splits them into sentences.
	Returns: 
	{
		potential1: [s1,s2,s3,.........]
		potential2:	[s1,s2,s3,.........]

	}
	'''

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



	'''
	Splits the input text into paragraphs: 
	Returns: [p1,p2,p3,......]
	Each paragraph comprises of 3 sentences (default)
	'''


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


	'''
	Splits the candidate books into paragraphs:
	Returns:
	{
		potential1:[p1,p2,p3,.................]
		potential2:[p1,p2,p3,p4....]
	}
	'''

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

	'''
	Divides paragraphs into chunks for multiprocessing at the later stages. The number of chunks is equal to the number of cores 
	Returns: [[p1,p2,p3,p4],[p5,p6,p7,p8],............]. 
	'''

	def splitChunks(self,textPara):
		n = self.cores
		num = float(len(textPara))/n
		l = [ textPara[i:i + int(num)] for i in range(0, (n-1)*int(num), int(num))]
		l.append(textPara[(n-1)*int(num):])
		return l	


	'''
	temporary function to perform basic processing using spacy. Allows for multilingual processing. Need to refactor. Currently returns the same data structures of textChunks and booksPara
	with spacy annotated data rather than the original data. 
	'''

	def spacyExtract(self,textChunks,books):
		
		# Choosing the correct spacy model based on the language: Add if statements for more language or automate the selection
		if self.language=='en':
			sp=sp_en
		if self.language=='de':
			sp=sp_de
		
		# 'new' text in spacy format and broken into chunks for multiprocessing
		spacyTextChunks=[]
		for chunk in textChunks:
			l=[]
			for para in chunk:
				l.append(sp(para))
			spacyTextChunks.append(l)

		# 'new' text in spacy format in a large list instead of list of lists. 
		spacyText=[]
		for chunk in spacyTextChunks:
			for sent in chunk:
				spacyText.append(sent)

		# 'potential' texts in spacy format
		spacyBooksPara=dict()
		for book in self.booksList:
			l=[]
			for para in books[book]:
				l.append(sp(para))
			spacyBooksPara[book]=l

		return spacyTextChunks,spacyBooksPara, spacyText

			
	'''
	A function which reduces the number of paragraphs present in the potential candidates. (New: Pass spacy text chunks and spacy paragraphs as parameters)
	Returns: 
	{
		potential1:[p1,p2,p5,......]
		potential2:[p2,p3,p7,..........]
	}

	New change: pass spacy text chunks and spacy paragraphs as parameters instead of the original text. 
	'''


	def filterWithJacard(self,textChunks,booksPara,threshold=0.15):

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
		
		# reduced paragraph numbers
		reducedPara=dict()
		for book in self.booksList:
			reducedPara[book]=list()		
		
		for para in jacardScores:
			for book in self.booksList:
				reducedPara[book]=reducedPara[book]+[x[1] for x in para[book]]
		
		for book in self.booksList:
			reducedPara[book]=list(set(reducedPara[book]))
		
		# the actual reduced paragraphs in spacy processed format
		reducedParagraphs=dict()
		for book in self.booksList:
			reducedParagraphs[book]=list()
	
		for book in self.booksList:
			for para in reducedPara[book]:
				reducedParagraphs[book].append(booksPara[book][para])
  		
		pool.close()
		return reducedParagraphs,reducedPara


	'''
	Temporary; refactor later. pass the original paragraphs and reduced paragraph numbers. Returns the reduced paragraphs
	'''

	def filterOriginalBooks(self,reducedPara,booksPara):
		reducedParagraphs=dict()
		for book in self.booksList:
			reducedParagraphs[book]=list()
	
		for book in self.booksList:
			for para in reducedPara[book]:
				reducedParagraphs[book].append(booksPara[book][para])
	
		return reducedParagraphs
		

	'''
	Syntactic parsing of the new book. 
	Returns 2 data structures.of the form:
	[chunk1,chuunk2,chunk3,.........]=[[p1,p2,p3,p4],[p5,p6,p7,p8],[......],..........]=

	[[[ptree1,ptree2,ptree3],[ptree2,ptree3,ptree4],[ptree3,ptree4,ptree5],[ptree4,ptree5,ptree6]],[p5,p6,p7,p8]]
	'''

	def parseNewBook(self,textChunks):
		# mapInput=[(textChunks[i],self.dependencies) for i in range(len(textChunks))]
		pool=Pool(processes=self.cores)
		results=pool.map(parseNewText,textChunks)
		parseTrees=list()
		parseWithoutTokenTrees=list()
		# parsedSentences=list()
		for i in range(len(results)):
			parseTrees.append(results[i][0])
			parseWithoutTokenTrees.append(results[i][1])
		pool.close()
		return parseTrees,parseWithoutTokenTrees

	'''
	Function to parse the candidates. Each candidate is processed on a separate core. 
	Returns: 2 dictionaryies of the following format:
	{
		potential1=[[ptree1,ptree2,ptree3],[ptree2,ptree3,ptree4],...........]
		potential2=[[ptree1,ptree2,ptree3],[ptree2,ptree3,ptree4],...........]
	}
	'''

	def parseCandidates(self,reducedParagraphs):
		booksToBeParsed=[reducedParagraphs[bk] for bk in self.booksList]
		pool=Pool(processes=len(reducedParagraphs))
		results=pool.map(parseCandidateBooks,booksToBeParsed)
		potentialParseTrees=dict()
		potentialParseWithoutTokenTrees=dict()
		i=0
		for bk in self.booksList:
			potentialParseTrees[bk]=results[i][0]
			potentialParseWithoutTokenTrees[bk]=results[i][1]
			i=i+1
		pool.close()
		return potentialParseTrees,potentialParseWithoutTokenTrees


	'''
	Returns 2 lists after computing the moschitti score between pairs of sentences. 
	The lists are in the following format: [dict1,dict2,dict3,dict4,............]
	Each dictionary is in the following format:
	{
		potential1:[0.1,0.5,....]
		potential2:[0.04,0.9,...]
	}

	The scores represent the average pairwise moschitti score between sentences from two paragraphs
	'''

	def syntacticScoring(self,parseTrees,potentialParseTrees,parseWithoutTokenTrees,potentialParseWithoutTokenTrees):
		
		# Scoring with tokens included
		mapInput=[(parseTrees[i],potentialParseTrees,self.booksList) for i in range(len(parseTrees))]
		pool=Pool(processes=self.cores)
		results=pool.map(scoreSyntax,mapInput)
		# print(len(results))
		syntacticScore=list()
		for scoreChunk in results:
			for score in scoreChunk:
				syntacticScore.append(score)
		# print('syntacticScore: ',len(syntacticScore))
		pool.close()

		print('Scoring without tokens')
		# Scoring without tokens included
		mapInput=[(parseWithoutTokenTrees[i],potentialParseWithoutTokenTrees,self.booksList) for i in range(len(parseWithoutTokenTrees))]
		pool=Pool(processes=self.cores)
		results=pool.map(scoreSyntax,mapInput)
		# print(len(results))
		syntacticScoreWithoutTokens=list()
		for scoreChunk in results:
			for score in scoreChunk:
				syntacticScoreWithoutTokens.append(score)
		# print('syntacticScore: ',len(syntacticScore))
		pool.close()
		return syntacticScore,syntacticScoreWithoutTokens

	'''	
	Caclualtes the semantic similairty between paragraphs from the new text and paragraphs in reduced books. No multiprocessing in this step yet. 
	Returns a list of dicitionaries. Each dictionary has the following format:

	{
		potential1:[(0.1,0.05,0.5,0.4,2),(),(),.........]
		potential2:[(0.4,0.15,0.4,0.1,0),(),(),.........]
		
	}, i.e. values are a list of tuples where each tuple has the following similarity metrics: semantic similarity, semantic similarity without stop words, semantic similarity of nouns, 
	semantic similarity of verbs, number of common proper nouns between the two paragraphs. 

	New change: Pass spacy data as parameters to this function rather than original text and books in paragraph format: text==SpacyText, reducedBooks==spacyReducedBooks

	'''
	
	def semanticScoring(self,textPara,reducedParagraphs,monolingual=True,lang1=None,lang2=None):
		if monolingual:
			semanticScore=list()
			for i in range(len(textPara)):
				scoreDict=dict()
				s1=avg_feature_vector(textPara[i],model,300,index2word_set)
				s1ws=avg_feature_vector_without_stopwords(textPara[i],model,300,index2word_set)
				s1n=avg_feature_vector_nouns(textPara[i],model,300,index2word_set)
				s1v=avg_feature_vector_verbs(textPara[i],model,300,index2word_set)
				for bk in self.booksList:
					df=list()
					for j in range(len(reducedParagraphs[bk])):
						s2=avg_feature_vector(reducedParagraphs[bk][j],model,300,index2word_set)
						s2ws=avg_feature_vector_without_stopwords(reducedParagraphs[bk][j],model,300,index2word_set)
						s2n=avg_feature_vector_nouns(reducedParagraphs[bk][j],model,300,index2word_set)
						s2v=avg_feature_vector_verbs(reducedParagraphs[bk][j],model,300,index2word_set)
						semScore=1 - spatial.distance.cosine(s1, s2)
						semScore_withoutStop=1 - spatial.distance.cosine(s1ws, s2ws)
						semScore_nouns=1 - spatial.distance.cosine(s1n, s2n)
						semScore_verbs=1 - spatial.distance.cosine(s1v, s2v)
						properNouns=commonProperNouns(textPara[i],reducedParagraphs[bk][j])
						df.append((semScore,semScore_withoutStop,semScore_nouns,semScore_verbs,properNouns))

					scoreDict[bk]=df
				semanticScore.append(scoreDict)
			return semanticScore
		# Refactor: automated language detection instead of if-else and parameters
		else:
			if lang1=='german':
				path1='/home/users2/mehrotsh/Downloads/wiki.multi.de.vec.txt'
			if lang1=='english':
				path1='/home/users2/mehrotsh/Downloads/wiki.multi.en.vec.txt'
			if lang2=='german':
				path2='/home/users2/mehrotsh/Downloads/wiki.multi.de.vec.txt'
			if lang2=='english':
				path2='/home/users2/mehrotsh/Downloads/wiki.multi.en.vec.txt'

			l1_embeddings, l1_id2word, l1_word2id = load_vec(path1)
			l2_embeddings, l2_id2word, l2_word2id = load_vec(path2)

			semanticScore=list()
			for i in range(len(textPara)):
				scoreDict=dict()
				s1=fasttext_avg_feature_vector(textPara[i],l1_embeddings,300,l1_word2id)
				s1ws=fasttext_avg_feature_vector_without_stopwords(textPara[i],l1_embeddings,300,l1_word2id)
				s1n=fasttext_avg_feature_vector_nouns(textPara[i],l1_embeddings,300,l1_word2id)
				s1v=fasttext_avg_feature_vector_verbs(textPara[i],l1_embeddings,300,l1_word2id)
				for bk in self.booksList:
					df=list()
					for j in range(len(reducedParagraphs[bk])):
						s2=fasttext_avg_feature_vector(reducedParagraphs[bk][j],l2_embeddings,300,l2_word2id)
						s2ws=fasttext_avg_feature_vector_without_stopwords(reducedParagraphs[bk][j],l2_embeddings,300,l2_word2id)
						s2n=fasttext_avg_feature_vector_nouns(reducedParagraphs[bk][j],l2_embeddings,300,l2_word2id)
						s2v=fasttext_avg_feature_vector_verbs(reducedParagraphs[bk][j],l2_embeddings,300,l2_word2id)
						semScore=1 - spatial.distance.cosine(s1, s2)
						semScore_withoutStop=1 - spatial.distance.cosine(s1ws, s2ws)
						semScore_nouns=1 - spatial.distance.cosine(s1n, s2n)
						semScore_verbs=1 - spatial.distance.cosine(s1v, s2v)
						properNouns=commonProperNouns(textPara[i],reducedParagraphs[bk][j])
						df.append((semScore,semScore_withoutStop,semScore_nouns,semScore_verbs,properNouns))
					scoreDict[bk]=df
				semanticScore.append(scoreDict)
			return semanticScore

	'''
	Calculates the longest common subsequence between paragraphs from new text and paragraphs from the reduced books. Returns:
	lcsScore: list of dictionaries: Each dictionary is of the following format: 
	{
		potential1:[3,0,2,...........]
		potential2:[0,2,1............] 

	}

	lcs: list of dictionaries: Each dictionary is of the following format: 
	{
		potential1:['so they human','','yes we',...........]
		potential2:['','we can','no',........]

	}
	'''



	def longestSubsequenceScoring(self,textPara,reducedParagraphs):
		lcs=list()
		lcsScore=list()
		for i in range(len(textPara)):
			scoreDict_lcs=dict()
			scoreDict_lcsScore=dict()
			for bk in self.booksList:
				df_lcs=list()
				df_lcsScore=list()
				for j in range(len(reducedParagraphs[bk])):
					data=[textPara[i],reducedParagraphs[bk][j]]
					subsequence=longestSubsequence(textPara[i],reducedParagraphs[bk][j])
					df_lcs.append(subsequence)
					df_lcsScore.append(len(subsequence.split()))
				scoreDict_lcs[bk]=df_lcs
				scoreDict_lcsScore[bk]=df_lcsScore
			lcs.append(scoreDict_lcs)
			lcsScore.append(scoreDict_lcsScore)
		return (lcsScore,lcs)

	'''
	A function to aggregate all the scoring mechanisms used till now. Returns a list of tuples. 
	Each tuple is in the following format: 
	(sentenceNumber, refBook, sentenceNumber in the ref,syntactic similarity, semantic similarity, semantic similarity without stopwords, semantic similarity nouns, semantic similarity verbs, average similairty,
	 lcs length, lcs, syntactic similarity without tokens, common proper nouns, jaccard nouns, jaccard verbs, jaccard adjectives)
	'''


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
	
	'''
	Every paragraph in the new text has a large list of tuples associated with it. These tuples are sorted in the order of number of common proper nouns and in the case of tie, using the average similarity. 
	If the first tuple in this pair has an avergae syntactic and semantic similarity greater than the threshhold, then it is added to the final filter of tuples. Currently, this mechanism chooses at most
	only one paragraph pair for every paragraph in the new text but this filtering mechanism can be changed easily. 
	'''

	def finalFiltering(self,scoreTuples,reducedParagraphs,threshold=0.89):
		totalPotentialSentences=0
		for bk in self.booksList:
			totalPotentialSentences=totalPotentialSentences+len(reducedParagraphs[bk])
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

		# Extracting tuples that have large differences in syntactic and semantic values and atleast one of them is greater than 0.8

		diffTuples=list()
		for tup in scoreTuples:
			if (tup[3]>0.8 and abs(tup[3]-tup[4])>=0.12) or (tup[4]>0.8 and abs(tup[3]-tup[4])>=0.12):
				diffTuples.append(tup)

		return finalTuples,diffTuples


	'''
	The final tuples are displayed in decreasing order of the jaccard of common nouns between the paragraphs. 

	New change: Pass spacy data rather than original textPara and reducedParagraphs. This is because new metrics are added in this step. Apart from the sorting. 
	'''

	def nounBasedRanking(self,finalTuples,textPara,reducedParagraphs):
		newTuples=list()
		for tup in finalTuples:
			originalSent=textPara[tup[0]]
			refSent=reducedParagraphs[tup[1]][tup[2]]
			nounScore=jacardNouns(originalSent,refSent)
			verbScore=jacardVerbs(originalSent,refSent)
			adjScore=jacardAdj(originalSent,refSent)
			newTuples.append(tup+(nounScore,verbScore,adjScore))
		newTuples.sort(key=itemgetter(13,8),reverse=True)
		return newTuples

	'''
	A function to write out the tuples along with the paragraphs into a file
	'''

	def writeOutput(self,newTuples,textPara,reducedParagraphs):
		f=open(self.outputFolder+'nounSortedSentencePairs.txt','w')
		i=1
		lines=list()
		for t in newTuples:
			j=str(i)
			lines.append('Pairing: '+j)
			lines.append('\n')
			lines.append('New Sentence: '+textPara[t[0]])
			lines.append('\n')
			lines.append('Reference: \n'+reducedParagraphs[t[1]][t[2]])
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


# A main function to demonstrate the use of the package

def main():

	# An object to detect allusions
	d=detectParagraph(inputFolder='../data/temp/',outputFolder='../output/temp/',cores=40)
	
	# Loading the data
	print('Loading books and splitting')
	text=d.loadNew()
	books=d.loadCandidates()

	# Converting the loaded texts into paragraphs
	textPara=d.splitNewPara(text)
	booksPara=d.splitCandidatesPara(books)

	# Converting into chunks for multiprocessing
	textChunks=d.splitChunks(textPara)

	# Processing the texts using spacy
	print('spacy')
	spacyTextChunks,spacyBooksPara,spacyText=d.spacyExtract(textChunks,booksPara)

	# Using the jaccarding coefficient to filter out sentences from the potential candidates
	print('Filtering using Jaccard')
	reducedSpacyBooks,reducedSentences=d.filterWithJacard(spacyTextChunks,spacyBooksPara,threshold=0.15) #filtering on spacy data structure
	reducedBooks=d.filterOriginalBooks(reducedSentences,booksPara) #filtering on the original books

	pickling_on = open('../output/'+'temp/reducedBooks.pickle',"wb")
	pickle.dump(reducedBooks, pickling_on)

	# Syntactic parsing of the new text
	print('Syntactic parsing')
	parseTrees,parseWithoutTokenTrees=d.parseNewBook(textChunks)

	pickling_on = open('../output/'+'temp/parseTrees.pickle',"wb")
	pickle.dump(parseTrees, pickling_on)

	# Syntactic parsing of the candidates
	potentialParseTrees,potentialParseWithoutTokenTrees=d.parseCandidates(reducedBooks)

	pickling_on = open('../output/'+'temp/potentialParseTrees.pickle',"wb")
	pickle.dump(potentialParseTrees, pickling_on)

	# Syntactic similarity scoring using the Moschitti score
	print('Moschitti scoring')
	syntacticScore,syntacticScoreWithoutTokens=d.syntacticScoring(parseTrees,potentialParseTrees,parseWithoutTokenTrees,potentialParseWithoutTokenTrees)

	pickling_on = open('../output/'+'temp/allScores.pickle',"wb")
	pickle.dump(syntacticScore, pickling_on)

	# Semantic scoring  
	print('Semantic scoring')
	semanticScore=d.semanticScoring(spacyText,reducedSpacyBooks,monolingual=True,lang1='english',lang2='english')

	# Extracting the longest common subsequence
	print('Extracting longest subsequence')
	lcsScore,lcs=d.longestSubsequenceScoring(textPara,reducedBooks)

	# Aggregating the semantic and syntactic scores
	print('Average scoring')
	scoreTuples=d.aggregateScoring(syntacticScore,semanticScore,lcsScore,lcs,syntacticScoreWithoutTokens)

	pickling_on = open('../output/'+'temp/scoreTuples.pickle',"wb")
	pickle.dump(scoreTuples, pickling_on)

	# Extracting a limited number of paragraph pairs
	finalTuples,diffTuples=d.finalFiltering(scoreTuples,reducedBooks,0.79)
	if len(finalTuples)>100:
		finalTuples=finalTuples[0:100]

	# Ranking the tuples using jaccard index of nouns
	orderedTuples=d.nounBasedRanking(finalTuples,spacyText,reducedSpacyBooks)
	
	pickling_on = open('../output/'+'temp/orderedTuples.pickle',"wb")
	pickle.dump(orderedTuples, pickling_on)

	# Printing out the final tuples on the terminal
	print('Final results: \n\n\n')

	i=1
	for t in orderedTuples:
		print('Pairing: ',i)
		print('\n')
		print('New Sentence: ',textPara[t[0]])
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
		print('Jaccard of common nouns: ',t[13])
		print('Jaccard of common verbs: ',t[14])
		print('Jaccard of common adjectives: ',t[15])
		print('Semantic similarity nouns: ',t[6])
		print('Semantic similarity verbs: ',t[7])
		print('\n\n')
		i=i+1

	# Writing the output into a file
	d.writeOutput(orderedTuples,textPara,reducedBooks)

	# Ranking the tuples with large differences in semantic and syntactic similarity
	diffTuples=d.nounBasedRanking(diffTuples,spacyText,reducedSpacyBooks)


if __name__=="__main__":
	main()





	