# DetectingAllusion
Computational detection of intertextual references and allusions. 

# Description
A python package which searches for potential allusions and references made by a book to a list of potential candidate books, i.e. books which are being referred to. 

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

# Initial Setup
* Download pre-trained google word2vec vectors 
* Download Stanford NLP 
# Variables and Data Structures that you will encounter 
* text: a list of sentences (new book): [sent1,sent2,sent3,.....]
* books: a dicitionary where the key is a name of potential book and the value is a list of sentences.
  {book1:[sent1,sent2,....];book2:[....];......}
* reducedBooks: similar to books with a reduced list of sentences 
* textChunks: a list of lists of sentences from text: [[sent1,sent2,sent3],[sent4,sent5,sent6],.....]
  The number of chunks is equal to the number of cores. Default is 40. 
* parseTrees: a list of lists of parseTrees. Each parse tree is in the form of a defaultdictionary. 
  [[parseTree1,parseTree2,parseTree3],[parseTree4,parseTree5,parseTree6],......]. There is a one to one mapping from textChunks to ParseTrees. 
* potentialParseTrees: A dicitonary where keys are names of potential books and the corresponding value is a list of parseTrees. 
  potentialParseTrees={book1: [parseTree1,parseTree2,....];book2:[.....]}. There is a one to one mapping between reducedBooks and       potentialParseTrees.
* syntacticScore: a list of dictionaries. [dict0,dict1,dict2.......]. Dict-i corresponds to sentence-i of text.
  Each dict looks something like this: dicti={book1:[0.2,0.5,.......];book2:[......]}, i.e. keys are book names of potential candidates and the corresponding value is a list comprising of moschitti scores between parseTrees[i] and potentialParseTrees[book]
  
 
  
# Usage

A typical pipeline would look like this:

* Load the text

```python
from detect import *
d=detect(inputFolder='../data/')
text=d.loadNew()
books=d.loadCandidates()
textChunks=d.splitChunks(text)
```
* Parse the books

```python
reducedBooks=d.filterWithJacard(textChunks,books,threshold=0.3)
parseTrees,parsedSentences=d.parseNewBook(textChunks)
potentialParseTrees,potentialParsedSentences=d.parseCandidates(reducedBooks)
```
* Syntactic and semantic scoring

```python
syntacticScore=d.syntacticScoring(parseTrees,potentialParseTrees)
semanticScore=d.semanticScoring(text,reducedBooks)
scoreTuples=d.aggeregateScoring(syntacticScore,semanticScore)
```
* Noun based ranking

```python
finalTuples=d.finalFiltering(scoreTuples,reducedBooks)
orderedTuples=d.nounBasedRanking(finalTuples,text,reducedBooks)
```
# License

MIT License

Copyright (c) 2018 Samarth Mehrotra

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


