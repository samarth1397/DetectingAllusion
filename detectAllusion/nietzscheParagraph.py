
from detectParagraph import *

def main():
	d=detectParagraph(inputFolder='../data/n1/',outputFolder='../output/n1/',cores=30)
	print('Loading books and splitting')
	text=d.loadNew()
	books=d.loadCandidates()

	textPara=d.splitNewPara(text)
	booksPara=d.splitCandidatesPara(books)

	textChunks=d.splitChunks(textPara)

	print('spacy')
	spacyTextChunks,spacyBooksPara=d.spacyExtract(textChunks,booksPara)

	print('Filtering using Jaccard')
	reducedSpacyBooks,reducedSentences=d.filterWithJacard(spacyTextChunks,spacyBooksPara,threshold=0.07)
	

	reducedBooks=d.filterOriginalBooks(reducedSentences,booksPara)

	pickling_on = open('../output/'+'n1/reducedBooks.pickle',"wb")
	pickle.dump(reducedBooks, pickling_on)


	print('Syntactic parsing')
	parseTrees,parseWithoutTokenTrees=d.parseNewBook(textChunks)
	pickling_on = open('../output/'+'n1/parseTrees.pickle',"wb")
	pickle.dump(parseTrees, pickling_on)


	potentialParseTrees,potentialParseWithoutTokenTrees=d.parseCandidates(reducedBooks)

	pickling_on = open('../output/'+'n1/potentialParseTrees.pickle',"wb")
	pickle.dump(potentialParseTrees, pickling_on)

	print('Moschitti scoring')
	syntacticScore,syntacticScoreWithoutTokens=d.syntacticScoring(parseTrees,potentialParseTrees,parseWithoutTokenTrees,potentialParseWithoutTokenTrees)

	pickling_on = open('../output/'+'n1/allScores.pickle',"wb")
	pickle.dump(syntacticScore, pickling_on)

	spacyText=[]
	for chunk in spacyTextChunks:
		for sent in chunk:
			spacyText.append(sent)

	print('Semantic scoring')
	semanticScore=d.semanticScoring(spacyText,reducedSpacyBooks,monolingual=True,lang1='english',lang2='english')

	print('Extracting longest subsequence')
	lcsScore,lcs=d.longestSubsequenceScoring(textPara,reducedBooks)

	print('Average scoring')
	scoreTuples=d.aggregateScoring(syntacticScore,semanticScore,lcsScore,lcs,syntacticScoreWithoutTokens)

	pickling_on = open('../output/'+'n1/scoreTuples.pickle',"wb")
	pickle.dump(scoreTuples, pickling_on)

	finalTuples,diffTuples=d.finalFiltering(scoreTuples,reducedBooks,0.82)
	if len(finalTuples)>100:
		finalTuples=finalTuples[0:100]

	orderedTuples=d.nounBasedRanking(finalTuples,spacyText,reducedSpacyBooks)
	
	pickling_on = open('../output/'+'n1/orderedTuples.pickle',"wb")
	pickle.dump(orderedTuples, pickling_on)

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

	# d.writeOutput(orderedTuples,text,reducedBooks)

	# print('\n\n Tuples with large difference in syntactic and semantic value: \n\n\n')

	# diffTuples=d.nounBasedRanking(diffTuples,textPara,reducedBooks)
	# d.writeOutput(diffTuples,textPara,reducedBooks)
	# pickling_on = open('../output/'+'n1/diffTuples.pickle',"wb")
	# pickle.dump(diffTuples, pickling_on)



if __name__=="__main__":
	main()