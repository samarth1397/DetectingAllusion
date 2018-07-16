from detect import * 

def main():
	d=detect(inputFolder='../data/n1/',outputFolder='../output/n1/',cores=30)
	print('Loading books and splitting')
	text=d.loadNew()
	books=d.loadCandidates()
	textChunks=d.splitChunks(text)

	print('spacy')
	spacyTextChunks,spacyBooks=d.spacyExtract(textChunks,books)

	print('Filtering using Jaccard')
	reducedSpacyBooks,reducedSentences=d.filterWithJacard(spacyTextChunks,spacyBooks,threshold=0.08)
	
	reducedBooks=d.filterOriginalBooks(reducedSentences,books)

	pickling_on = open('../output/'+'n1/reducedBooks.pickle',"wb")
	pickle.dump(reducedBooks, pickling_on)


	# print('Text: ',len(text))
	# print('original is',len(books['isaiah']))
	# print('reduced isaiah',len(reducedBooks['isaiah']))

	# print('textChunks: ',len(textChunks))


	print(len(text))

	for key in reducedSpacyBooks.keys():
		print(key,len(reducedSpacyBooks[key]))


	print('Syntactic parsing')
	parseTrees,parsedSentences,parseWithoutTokenTrees=d.parseNewBook(textChunks)
	pickling_on = open('../output/'+'n1/parseTrees.pickle',"wb")
	pickle.dump(parseTrees, pickling_on)

	# print('Parse trees',len(parseTrees))

	potentialParseTrees,potentialParsedSentences,potentialParseWithoutTokenTrees=d.parseCandidates(reducedBooks)
	# print(len(parseTrees))
	# print(len(parseTrees['isaiah']))
	pickling_on = open('../output/'+'n1/potentialParseTrees.pickle',"wb")
	pickle.dump(potentialParseTrees, pickling_on)

	# print('Potential Parse Trees isaiah ',len(potentialParseTrees['isaiah']))

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

	# print('Semantic Score: ',len(semanticScore))

	print('Extracting longest subsequence')
	lcsScore,lcs=d.longestSubsequenceScoring(text,reducedBooks)

	print('Average scoring')

	scoreTuples=d.aggregateScoring(syntacticScore,semanticScore,lcsScore,lcs,syntacticScoreWithoutTokens)

	# print(len(scoreTuples))


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

	d.writeOutput(orderedTuples,text,reducedBooks)

	print('\n\n Tuples with large difference in syntactic and semantic value: \n\n\n')

	# diffTuples=d.nounBasedRanking(diffTuples,text,reducedBooks)
	# d.writeOutput(diffTuples,text,reducedBooks)
	# pickling_on = open('../output/'+'n1/diffTuples.pickle',"wb")
	# pickle.dump(diffTuples, pickling_on)
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
	
		
		