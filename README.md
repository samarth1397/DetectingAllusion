# DetectingAllusion
Computational detection of intertextual references

# Description
A python package which searches for potential allusions and intertextual references made in a book against a set of potential candidates. 

# Terminology
* New Text: The book which might be making allusions to other books
* Potential Candidates: The books to which the references might have been made

# Methodology
* Every sentence in the new text is compared with every other sentence in the potential candidates for structural and semantic similarity. 
* Structural similarity is calculated using the Moschitti Score. 
* Semantic similarity is calculated using the cosine similarity between the average word vectors of two sentences. Pre-trained google word2vec vectors are used. 
* In order to speed up the entire process, we initially discard sentences from the potential candidates as follows: 
If a sentence in the potential candidates does not have a minimum (user-defined) jaccardian index (word overlap) with any of the sentences in the new text, then this sentence is discarded right at the start, i.e. before the semnatic and similarity checks. 
* We then order the pairs of sentences(sentence from new text paired with a sentence from the potential candidates) based on their average similairty and discard of all the pairs that are below the threshold (again, user-defined). 
* Finally, we rank the left over sentence pairs based on the jaccard index of nouns in the two sentences. This is because increased presence of nouns indicates higher chances of actual allusions and ranks these sentences above the potential false positives. 

# Usage

A typical pipeline would look like this:

```python
from detect import *
d=detect(inputFolder='../data/')
text=d.loadNew()
books=d.loadCandidates()
textChunks=d.splitChunks(text)
reducedBooks=d.filterWithJacard(textChunks,books,threshold=0.3)
parseTrees,parsedSentences=d.parseNewBook(textChunks)
potentialParseTrees,potentialParsedSentences=d.parseCandidates(reducedBooks)
syntacticScore=d.syntacticScoring(parseTrees,potentialParseTrees)
semanticScore=d.semanticScoring(text,reducedBooks)
scoreTuples=d.aggeregateScoring(syntacticScore,semanticScore)
finalTuples=d.finalFiltering(scoreTuples,reducedBooks)
orderedTuples=d.nounBasedRanking(finalTuples,text,reducedBooks)
```

