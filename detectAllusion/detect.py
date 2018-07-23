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
import io
from treeFunctions import *
from dependencies import *

	
'''
A class which contains all the methods required for detecting sentence-level allusions between an input text and a list of candidates
'''

class detect:

	'''
	Initialize input folder, output folder, path to dependencies, language for processing and number of available cores
	'''
	def __init__(self,inputFolder='../data/',outputFolder='../output/',dependencies='/home/users2/mehrotsh/scripts/packages/',language='en',cores=40):
		self.potential=inputFolder+'potential/'
		self.new=inputFolder+'new/'
		self.pickled=outputFolder+'pickle/'
		self.booksList=os.listdir(self.potential)
		self.dependencies=dependencies+'stanford-corenlp-full-2018-02-27/'
		# self.books=dict()
		self.cores=cores
		self.output=outputFolder
		self.language=language
	
	'''
	Loads the new text from the input folder. Returns a list of sentences. 
	'''

	def loadNew(self):
		test=os.listdir(self.new)[0]
		testB=open(self.new+test)
		raw=testB.read()
		text = strip_headers(raw).strip()
		text=text.replace('\n',' ')
		text=text.replace(':','. ')
		text=sent_tokenize(text)
		text = list(filter(lambda x: len(x)>11, text))	
		return text
		
	'''
	A function to load the candidates from the input folder. 
	Returns a dictionary where keys are the names of candidates and the corresponding value is the list of sentences of the candidate
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
			candidate = list(filter(lambda x: len(x)>11, candidate))
			books[file]=candidate
		return books		
	
	'''
	Temporary function: discards the extremely long sentences from the new text which might cause timeout errors during parsing. 
	'''

	def discardLongSentencesNew(self,text,maxLen=100):
		i=0
		for sent in text:
			if len(word_tokenize(sent))>maxLen:
				i=i+1
				text.remove(sent)
		print('Number of discarded sentences=: ',i+1)
		print('Number of leftover sentences in text: ',len(text))
		return text

	'''	
	Split the new book into chunks. The number of chunks is equal to the number of cores. 
	Returns a list of lists of sentences:
	l=[[s1,s2,s3],[s4,s5,s6],[....],..................]
	'''
		
	def splitChunks(self,text):
		n = self.cores
		num = float(len(text))/n
		l = [ text[i:i + int(num)] for i in range(0, (n-1)*int(num), int(num))]
		l.append(text[(n-1)*int(num):])
		return l	
	
	'''
	temporary function to perform basic processing using spacy. Returns the text chunks and books in the same data structure. 
	spacyTextChunks=[[s1,s2,s3],[s4,s5,s6],[....],..................]
	spacyBooks=
	{
		potential1=[s1,s2,s3,............]
		potentail2=[s1,s2,......]	

	}
	where each 's' is the sentence after spacy processing. Each token's lemma and pos tag has been identified. 

	'''

	def spacyExtract(self,textChunks,books):

		# Choosing the correct spacy model based on the language: Add if statements for more language or automate the selection
		if self.language=='en':
			sp=sp_en
		if self.language=='de':
			sp=sp_de

		spacyTextChunks=[]
		for chunk in textChunks:
			l=[]
			for sent in chunk:
				l.append(sp(sent))
			spacyTextChunks.append(l)


		# 'new' text in spacy format in a large list instead of list of lists. 
		spacyText=[]
		for chunk in spacyTextChunks:
			for sent in chunk:
				spacyText.append(sent)

		spacyBooks=dict()
		for book in self.booksList:
			l=[]
			for sent in books[book]:
				l.append(sp(sent))
			spacyBooks[book]=l

		return spacyTextChunks,spacyBooks,spacyText



	'''		
	Uses the calcJacardChunk function from treefunctions.py. Each chunk of the input text is processed on a new core. The threshold decided the minimum jaccard coefficient for 
	discarding sentences from the candidates. 
	Returns the same data structure of spacy books with reduced lists in the dictionary.

	New change: pass spacy text chunks and spacy books as parameters instead of the original text. 
	'''

	def filterWithJacard(self,textChunks,books,threshold=0.35):

		mapInput=[(textChunks[i],books,self.booksList) for i in range(len(textChunks))]
	
		pool=Pool(processes=self.cores)
		results=pool.map(calcJacardChunk,mapInput)
		
		jacardScores=[]
		for scoreChunk in results:
			for score in scoreChunk:
				jacardScores.append(score)

		for sent in jacardScores:
			for book in self.booksList:
				sent[book]=list(filter(lambda tup: tup[0]>threshold,sent[book]))
		
		# a dictionary where the keys are potential book names, and values are sentence numbers
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
		return reducedBooks,reducedSentences

	'''
	Temporary; refactor later; Currently, uses the reducedSentences data structure to filter the books data structure. 
	'''

	def filterOriginalBooks(self,reducedSentences,books):
		reducedBooks=dict()
		for book in self.booksList:
			reducedBooks[book]=list()
	
		for book in self.booksList:
			for sent in reducedSentences[book]:
				reducedBooks[book].append(books[book][sent])
	
		return reducedBooks
		
	'''
	Temporary: A function to extend the stopwords list to ignore very common proper nouns from the text, for example 'God'
	'''

	def extendStopwords(self,text):
		t=' '.join(text)
		t=tokenizer.tokenize(t)
		fdist=nltk.FreqDist(t)
		words=fdist.most_common(100)
		words=[word for word,score in words]
		tags=nltk.pos_tag(words)
		propNouns=[word for word,tag in tags if tag=='NNP']
		stopwords.extend(propNouns)
	'''
	Yet to be implemented
	'''

	def filterWithTFIDF(self):
		pass
	
	
	'''
	Generates parse trees for all the sentences in the new text. Each text chunk is processed on a separate core. Returns 3 list of lists of trees. 
	parseTrees: [[parseTree1,parseTree2,parseTree3],[parseTree4,parseTree5,parseTree6].............]
	'''

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
	
	'''
	Generates parse trees for the candidate books. Each book is processed on a separate core. Returns 3 dictionaries of parse Trees. 
	Keys of the dictionaries are names of the books and values are lists of parse trees. 
	'''
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

	'''
	Calcualtes the moschiiti score between sentences in textChunks and the sentences in the candidates. Returns 2 list of dictionaries. 
	Each dictionary is in the following format: 
	{
		potential1:[0.1,0.5,....]
		potential2:[0.04,0.9,...]
	}

	'''

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
			
	'''
	Caclualtes the semantic similairty between sentences from the new text and sentences in reduced books. No multiprocessing in this step yet. 
	Returns a list of dicitionaries. Each dictionary has the following format:

	{
		potential1:[(0.1,0.05,0.5,0.4,2),(),(),.........]
		potential2:[(0.4,0.15,0.4,0.1,0),(),(),.........]
		
	}, i.e. values are a list of tuples where each tuple has the following similarity metrics: semantic similarity, semantic similarity without stop words, semantic similarity of nouns, 
	semantic similarity of verbs, number of common proper nouns between the two sentences. 


	New change: Pass spacy data as parameters to this function rather than original text and reducedBooks: text==SpacyText, reducedBooks==spacyReducedBooks

	'''

	def semanticScoring(self,text,reducedBooks,monolingual=True,lang1=None,lang2=None):
		if monolingual:
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
		else:
			# refactor to automated language detection and loading of vectors

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
			for i in range(len(text)):
				scoreDict=dict()
				s1=fasttext_avg_feature_vector(text[i],l1_embeddings,300,l1_word2id)
				s1ws=fasttext_avg_feature_vector_without_stopwords(text[i],l1_embeddings,300,l1_word2id)
				s1n=fasttext_avg_feature_vector_nouns(text[i],l1_embeddings,300,l1_word2id)
				s1v=fasttext_avg_feature_vector_verbs(text[i],l1_embeddings,300,l1_word2id)
				for bk in self.booksList:
					df=list()
					for j in range(len(reducedBooks[bk])):
						s2=fasttext_avg_feature_vector(reducedBooks[bk][j],l2_embeddings,300,l2_word2id)
						s2ws=fasttext_avg_feature_vector_without_stopwords(reducedBooks[bk][j],l2_embeddings,300,l2_word2id)
						s2n=fasttext_avg_feature_vector_nouns(reducedBooks[bk][j],l2_embeddings,300,l2_word2id)
						s2v=fasttext_avg_feature_vector_verbs(reducedBooks[bk][j],l2_embeddings,300,l2_word2id)
						semScore=1 - spatial.distance.cosine(s1, s2)
						semScore_withoutStop=1 - spatial.distance.cosine(s1ws, s2ws)
						semScore_nouns=1 - spatial.distance.cosine(s1n, s2n)
						semScore_verbs=1 - spatial.distance.cosine(s1v, s2v)
						properNouns=commonProperNouns(text[i],reducedBooks[bk][j])
						df.append((semScore,semScore_withoutStop,semScore_nouns,semScore_verbs,properNouns))
					scoreDict[bk]=df
				semanticScore.append(scoreDict)
			return semanticScore




	'''
	Calculates the longest common subsequence between sentences from new text and sentences from the reduced books. Returns:
	lcsScore: list of dictionaries: Each dictionary is of the following format: 
	{
		potential1:[3,0,2,...........]
		potential2:[0,2,1............] 

	}

	lcs: list of dictionaries: Each dictionary is of the following format: 
	{
		potential1:[s'o they human','','yes we',...........]
		potential2:['','we can','no',........]

	}

	Note: pass original text and not spacy processed text as parameters
	'''

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

	'''
	A function to aggregate all the scoring mechanisms used till now. Returns a list of tuples. 
	Each tuple is in the following format: 
	(sentenceNumber, refBook, sentenceNumber in the ref,syntactic similarity, semantic similarity, semantic similarity without stopwords, semantic similarity nouns, semantic similarity verbs, average similairty, lcs length, lcs, 
	syntactic similarity without tokens, common proper nouns), 
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
	Every sentence in the new text has a large list of tuples associated with it. These tuples are sorted in the order of number of common proper nouns and in the case of tie, using the average similarity. 
	If the first tuple in this pair has an avergae syntactic and semantic similarity greater than the threshhold, then it is added to the final filter of tuples. Currently, this mechanism chooses at most
	only one sentence pair for every sentence in the new text but this filtering mechanism can be changed easily. 
	'''


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

		# identifying tuples which have large difference in semantic and syntactic similarity and atleast one of semantic or semantic similarity is greater than 0.8

		diffTuples=list()
		for tup in scoreTuples:
			if (tup[3]>0.8 and abs(tup[3]-tup[4])>=0.12) or (tup[4]>0.8 and abs(tup[3]-tup[4])>=0.12):
				diffTuples.append(tup)

		return finalTuples,diffTuples

	'''
	The final tuples are displayed in decreasing order of the jaccard of common nouns between the sentences. jaccard nouns, jaccard verbs, jaccard adjectives)

	New change: Pass spacy data rather than original text and reducedBooks. This is because a few more metrics are being calculated in this step which use the spacy processed text. 

	'''
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


	'''
	A function to write out the tuples along with the sentences into a file. Pass the original text and reduced books as parameters. 
	'''

	def writeOutput(self,newTuples,text,reducedBooks,fileName='nounSortedSentencePairs.txt'):
		f=open(self.output+fileName,'w')
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

# A main function to demonstrate the use of the package

def main():

	# Creating an object to detect sentence level allusions
	d=detect(inputFolder='../data/temp/',outputFolder='../output/temp/',cores=30)
	
	# loading the data
	print('Loading books and splitting')
	text=d.loadNew()
	books=d.loadCandidates()
	textChunks=d.splitChunks(text)

	# d.extendStopwords(text)
	
	# processing using spacy
	print('spacy')
	spacyTextChunks,spacyBooks,spacyText=d.spacyExtract(textChunks,books)

	# filtering using jaccard
	print('Filtering using Jaccard')
	reducedSpacyBooks,reducedSentences=d.filterWithJacard(spacyTextChunks,spacyBooks,threshold=0.2) #filtering the spacy data structure
	reducedBooks=d.filterOriginalBooks(reducedSentences,books) #filtering the original data structure

	pickling_on = open('../output/'+'temp/reducedBooks.pickle',"wb")
	pickle.dump(reducedBooks, pickling_on)

	# Syntactic parsing of the new text
	print('Syntactic parsing')
	parseTrees,parsedSentences,parseWithoutTokenTrees=d.parseNewBook(textChunks)

	pickling_on = open('../output/'+'temp/parseTrees.pickle',"wb")
	pickle.dump(parseTrees, pickling_on)

	# Syntactic parsing of the potential candidates
	potentialParseTrees,potentialParsedSentences,potentialParseWithoutTokenTrees=d.parseCandidates(reducedBooks)

	pickling_on = open('../output/'+'temp/potentialParseTrees.pickle',"wb")
	pickle.dump(potentialParseTrees, pickling_on)

	# Syntactic scoring using the moschitti score
	print('Moschitti scorings')
	syntacticScore,syntacticScoreWithoutTokens=d.syntacticScoring(parseTrees,potentialParseTrees,parseWithoutTokenTrees,potentialParseWithoutTokenTrees)

	pickling_on = open('../output/'+'temp/allScores.pickle',"wb")
	pickle.dump(syntacticScore, pickling_on)

	# Semantic scoring using word2vec
	print('Semantic scoring')
	semanticScore=d.semanticScoring(spacyText,reducedSpacyBooks,monolingual=True,lang1='english',lang2='english')

	# Extracting the longest common subsequence
	print('Extracting longest subsequence')
	lcsScore,lcs=d.longestSubsequenceScoring(text,reducedBooks)

	# Aggregating the syntactic and semantic scores
	print('Average scoring')
	scoreTuples=d.aggregateScoring(syntacticScore,semanticScore,lcsScore,lcs,syntacticScoreWithoutTokens)

	pickling_on = open('../output/'+'temp/scoreTuples.pickle',"wb")
	pickle.dump(scoreTuples, pickling_on)

	# Extracting a limited number of sentence pairs
	finalTuples,diffTuples=d.finalFiltering(scoreTuples,reducedBooks,0.79)
	if len(finalTuples)>100:
		finalTuples=finalTuples[0:100]
	
	# Sorting the extracted tuples using Noun based ranking
	orderedTuples=d.nounBasedRanking(finalTuples,spacyText,reducedSpacyBooks)
	
	pickling_on = open('../output/'+'temp/orderedTuples.pickle',"wb")
	pickle.dump(orderedTuples, pickling_on)


	# Printing final results on the terminal
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
		print('Jaccard of common nouns: ',t[13])
		print('Jaccard of common verbs: ',t[14])
		print('Jaccard of common adjectives: ',t[15])
		print('Semantic similarity nouns: ',t[6])
		print('Semantic similarity verbs: ',t[7])
		print('\n\n')
		i=i+1

	# Writing the output into a file
	d.writeOutput(orderedTuples,text,reducedBooks)

	# Sorting the tuples which had high differences

	diffTuples=d.nounBasedRanking(diffTuples,spacyText,reducedSpacyBooks)

	pickling_on = open('../output/'+'temp/diffTuples.pickle',"wb")
	pickle.dump(diffTuples, pickling_on)



if __name__=="__main__":
	main()
	
		
		
		
		
