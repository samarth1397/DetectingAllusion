
from detectParagraph import *

# A main function to demonstrate the use of the package

def main():

	# An object to detect allusions
	d=detectParagraph(inputFolder='../data/n4/',outputFolder='../output/n4/',cores=32)
	
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
	reducedSpacyBooks,reducedSentences=d.filterWithJacard(spacyTextChunks,spacyBooksPara,threshold=0.05) #filtering on spacy data structure
	reducedBooks=d.filterOriginalBooks(reducedSentences,booksPara) #filtering on the original books

	pickling_on = open('../output/'+'n4/reducedBooks.pickle',"wb")
	pickle.dump(reducedBooks, pickling_on)

	# Syntactic parsing of the new text
	print('Syntactic parsing')
	parseTrees,parseWithoutTokenTrees=d.parseNewBook(textChunks)

	pickling_on = open('../output/'+'n4/parseTrees.pickle',"wb")
	pickle.dump(parseTrees, pickling_on)

	# Syntactic parsing of the candidates
	potentialParseTrees,potentialParseWithoutTokenTrees=d.parseCandidates(reducedBooks)

	pickling_on = open('../output/'+'n4/potentialParseTrees.pickle',"wb")
	pickle.dump(potentialParseTrees, pickling_on)

	# Syntactic similarity scoring using the Moschitti score
	print('Moschitti scoring')
	syntacticScore,syntacticScoreWithoutTokens=d.syntacticScoring(parseTrees,potentialParseTrees,parseWithoutTokenTrees,potentialParseWithoutTokenTrees)

	pickling_on = open('../output/'+'n4/allScores.pickle',"wb")
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

	pickling_on = open('../output/'+'n4/scoreTuples.pickle',"wb")
	pickle.dump(scoreTuples, pickling_on)

	# Extracting a limited number of paragraph pairs
	finalTuples,diffTuples=d.finalFiltering(scoreTuples,reducedBooks,0.79)
	if len(finalTuples)>100:
		finalTuples=finalTuples[0:100]

	# Ranking the tuples using jaccard index of nouns
	orderedTuples=d.nounBasedRanking(finalTuples,spacyText,reducedSpacyBooks)
	
	pickling_on = open('../output/'+'n4/orderedTuples.pickle',"wb")
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
		print('Jaccard of common nouns: ',t[12])
		print('Jaccard of common verbs: ',t[13])
		print('Jaccard of common adjectives: ',t[14])
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






