from detect import *

def main():

	# Creating an object to detect sentence level allusions
	d=detect(inputFolder='../data/n3-lim/',outputFolder='../output/n3-lim-sent/',cores=30,language='de')
	
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
	reducedSpacyBooks,reducedSentences=d.filterWithJacard(spacyTextChunks,spacyBooks,threshold=0.05) #filtering the spacy data structure
	reducedBooks=d.filterOriginalBooks(reducedSentences,books) #filtering the original data structure

	pickling_on = open('../output/'+'n3-lim-sent/reducedBooks.pickle',"wb")
	pickle.dump(reducedBooks, pickling_on)

	# Syntactic parsing of the new text
	print('Syntactic parsing')
	parseTrees,parsedSentences,parseWithoutTokenTrees=d.parseNewBook(textChunks)

	pickling_on = open('../output/'+'n3-lim-sent/parseTrees.pickle',"wb")
	pickle.dump(parseTrees, pickling_on)

	# Syntactic parsing of the potential candidates
	potentialParseTrees,potentialParsedSentences,potentialParseWithoutTokenTrees=d.parseCandidates(reducedBooks)

	pickling_on = open('../output/'+'n3-lim-sent/potentialParseTrees.pickle',"wb")
	pickle.dump(potentialParseTrees, pickling_on)

	# Syntactic scoring using the moschitti score
	print('Moschitti scorings')
	syntacticScore,syntacticScoreWithoutTokens=d.syntacticScoring(parseTrees,potentialParseTrees,parseWithoutTokenTrees,potentialParseWithoutTokenTrees)

	pickling_on = open('../output/'+'n3-lim-sent/allScores.pickle',"wb")
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

	pickling_on = open('../output/'+'n3-lim-sent/scoreTuples.pickle',"wb")
	pickle.dump(scoreTuples, pickling_on)

	# Extracting a limited number of sentence pairs
	finalTuples,diffTuples=d.finalFiltering(scoreTuples,reducedBooks,0.79)
	if len(finalTuples)>100:
		finalTuples=finalTuples[0:100]
	
	# Sorting the extracted tuples using Noun based ranking
	orderedTuples=d.nounBasedRanking(finalTuples,spacyText,reducedSpacyBooks)
	
	pickling_on = open('../output/'+'n3-lim-sent/orderedTuples.pickle',"wb")
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

	pickling_on = open('../output/'+'n3-lim-sent/diffTuples.pickle',"wb")
	pickle.dump(diffTuples, pickling_on)



if __name__=="__main__":
	main()
	
		
		
		
		
