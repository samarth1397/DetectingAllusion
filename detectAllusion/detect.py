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



class detect:
	def __init__(self,inputFolder='../data/',OutputFolder='../output/',dependenciesLocation='/home/users2/mehrotsh/scripts/packages/stanford-corenlp-full-2018-02-27/',language='english'):
		potential=inputFolder+'potential/'
		newText=inputFolder+'new/'
		pickled=outputFolder+'pickle/'
		
		return 	
	
	def loadNew():
		pass
	
	def loadCandidates():
		pass
	
	def filterWithJacard():
		pass
	
	
	def filterWithTFIDF():
		pass
	
	
	def parseNewBook():
		pass
	
	
	def parseCandidates():
		pass
		
		
	def syntacticScoring():
		pass
		
	def semanticScoring():
		pass
		
	def aggeregateScoring():
		pass
		
	def finalFiltering():
		pass
		
		
		
		
		
