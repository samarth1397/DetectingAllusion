# DetectingAllusion
Computational detection of intertextual references and allusions. 

# Description
A python package which searches for potential allusions and references made by a book to a list of potential candidate books, i.e. books which are being referred to. 

# Terminology
* New Text: The book which might be making allusions to other books
* Potential Candidates: The books to which the references might have been made

# Overview of Methodology
* Every sentence in the new text is compared with every other sentence in the potential candidates for structural and semantic similarity. 
* Structural similarity is calculated using the Moschitti Score. 
* Semantic similarity is calculated using the cosine similarity between the average word vectors of two sentences. Pre-trained google word2vec vectors are used. 
* In order to speed up the entire process, we initially discard sentences from the potential candidates as follows: 
If a sentence in the potential candidates does not have a minimum (user-defined) jaccardian index (word overlap) with any of the sentences in the new text, then this sentence is discarded right at the start, i.e. before the semnatic and similarity checks. 
* We then order the pairs of sentences(sentence from new text paired with a sentence from the potential candidates) based on their average similairty and discard of all the pairs that are below the threshold (again, user-defined). 
* Finally, we rank the left over sentence pairs based on the jaccard index of nouns in the two sentences. This is because increased presence of nouns indicates higher chances of actual allusions and ranks these sentences above the potential false positives. 
* Currently, the system extracts a number of semantic and syntactic similarity metrics and displays them to the user in the final output. However, the system only uses the Moschitti score and cosine of word2vec to filter out the tuples. 

# Initial Setup
* Download pre-trained google word2vec vectors 
* Download Stanford NLP and the packages for the languages that you want to process
* Download spacy models for the languages that you want to process
* You might also have to download certain modules from NLTK which are not downloaded automatically. Hence, you might get some errors when you are running it for the first time. 

  
# Usage

Before you use the scripts, follow the instructions in the dependencies.py file to select the correct language and model. In the future, we will automate this step. 

#### Sentence level comparisons

A typical pipeline would look like this, if you want to detect allusions by comparing sentences:

* Load the text

```python
from detect import *
d=detect(inputFolder='../data/',outputFolder='../output/',cores=30)
text=d.loadNew()
books=d.loadCandidates()
textChunks=d.splitChunks(text)
```
* Process the texts using Spacy
```python
spacyTextChunks,spacyBooks,spacyText=d.spacyExtract(textChunks,books)
```

* Filter out books using the Jaccardian Index
```python
reducedSpacyBooks,reducedSentences=d.filterWithJacard(spacyTextChunks,spacyBooks,threshold=0.3) #filtering the spacy data structure
reducedBooks=d.filterOriginalBooks(reducedSentences,books)
```

* Parse the books

```python
parseTrees,parsedSentences,parseWithoutTokenTrees=d.parseNewBook(textChunks)
potentialParseTrees,potentialParsedSentences,potentialParseWithoutTokenTrees=d.parseCandidates(reducedBooks)
```
* Extracting similarity metrics: Syntactic, Semantic, Longest Common Subsequence

```python
syntacticScore,syntacticScoreWithoutTokens=d.syntacticScoring(parseTrees,potentialParseTrees,parseWithoutTokenTrees,potentialParseWithoutTokenTrees)
semanticScore=d.semanticScoring(spacyText,reducedSpacyBooks,monolingual=True)
lcsScore,lcs=d.longestSubsequenceScoring(text,reducedBooks)
scoreTuples=d.aggregateScoring(syntacticScore,semanticScore,lcsScore,lcs,syntacticScoreWithoutTokens)
```
* Final Filtering and Noun based ranking

```python
finalTuples,diffTuples=d.finalFiltering(scoreTuples,reducedBooks,0.75)
orderedTuples=d.nounBasedRanking(finalTuples,spacyText,reducedSpacyBooks)
```

* Write out the sentence pairs 	to a file

```python
d.writeOutput(orderedTuples,text,reducedBooks)
```
#### Paragraph level comparisons

A typical pipeline would look like this: 

* Load the text

```python 
from detect import *
d=detect(inputFolder='../data/',outputFolder='../output/',cores=30)
text=d.loadNew()
books=d.loadCandidates()
textPara=d.splitNewPara(text)
booksPara=d.splitCandidatesPara(books)
textChunks=d.splitChunks(textPara)
```
* Process the texts using Spacy
```python
spacyTextChunks,spacyBooksPara,spacyText=d.spacyExtract(textChunks,booksPara)
```

* Filter out books using the Jaccardian Index
```python
reducedSpacyBooks,reducedSentences=d.filterWithJacard(spacyTextChunks,spacyBooksPara,threshold=0.3) #filtering on spacy data structure
reducedBooks=d.filterOriginalBooks(reducedSentences,booksPara)
```

* Parse the books

```python
parseTrees,parseWithoutTokenTrees=d.parseNewBook(textChunks)
otentialParseTrees,potentialParseWithoutTokenTrees=d.parseCandidates(reducedBooks)
```
* Extracting similarity metrics: Syntactic, Semantic, Longest Common Subsequence

```python
syntacticScore,syntacticScoreWithoutTokens=d.syntacticScoring(parseTrees,potentialParseTrees,parseWithoutTokenTrees,potentialParseWithoutTokenTrees)
semanticScore=d.semanticScoring(spacyText,reducedSpacyBooks,monolingual=True)
lcsScore,lcs=d.longestSubsequenceScoring(textPara,reducedBooks)
scoreTuples=d.aggregateScoring(syntacticScore,semanticScore,lcsScore,lcs,syntacticScoreWithoutTokens)
```
* Final Filtering and Noun based ranking

```python
finalTuples,diffTuples=d.finalFiltering(scoreTuples,reducedBooks,0.75)
orderedTuples=d.nounBasedRanking(finalTuples,spacyText,reducedSpacyBooks)
```

* Write out the paragraph pairs to a file

```python
d.writeOutput(orderedTuples,textPara,reducedBooks)
```



# Potential Improvements
* Better class design to process paragraphs and sentences using a single class. (The sentence level processing is redundant since setting the number of sentences in
a paragraph to 1 is essentiall the same thing). 
* Automated language detection before semantic scoring and while processing the texts using Stanford NLP and Spacy
* New syntactic similarity metrics for cross language comparison
* Change the Jaccard Index approach to a better approach. (Maybe Filtering using TF-IDF might work)
* Change the final filtering to some form of a classification task: Allusion or not, based on the metrics that have been extracted. 

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


