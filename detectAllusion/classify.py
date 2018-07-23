
import numpy as np
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load true allusions
pickle_off = open("../output/nietzsche/orderedTuples.pickle","rb")
scoreTuples = pickle.load(pickle_off)

trueAllusions=list()
for tup in scoreTuples:
    trueAllusions.append(list(tup))


# Load false allusions
pickle_off = open("../output/n1-lim//orderedTuples.pickle","rb")
scoreTuples_1 = pickle.load(pickle_off)


pickle_off = open("../output/n3-lim//orderedTuples.pickle","rb")
scoreTuples_2 = pickle.load(pickle_off)


falseAllusions=list()
for tup in scoreTuples_1:
    falseAllusions.append(list(tup)[3:])
for tup in scoreTuples_2:
    falseAllusions.append(list(tup)[3:])



# Converting to numpy arrays
tr=np.array(trueAllusions)
fa=np.array(falseAllusions)
tr=np.delete(tr,7,axis=1) # remove LCS string
fa=np.delete(fa,7,axis=1) # remove LCS string
tr=tr.astype(float)
tr=np.nan_to_num(tr)
fa=fa.astype(float)
fa=np.nan_to_num(fa)

# Write basic statistics of metrics to a file
metrics=list()

metrics.append(list(np.amin(tr,0)))
metrics.append(list(np.amax(tr,0)))
metrics.append(list(np.mean(tr,0)))
metrics.append(list(np.median(tr,0)))
metrics.append(list(np.amin(fa,0)))
metrics.append(list(np.amax(fa,0)))
metrics.append(list(np.mean(fa,0)))
metrics.append(list(np.median(fa,0)))

colNames=['syntactic similarity', 'semantic similarity all words', 'semantic similarity without stopwords', 
          'semantic similarity nouns', 'semantic similarity verbs', 'average similairty',
          'lcs length', 'syntactic similarity without tokens', 'common proper nouns', 'jaccard nouns', 
          'jaccard verbs', 'jaccard adjectives']

df=pd.DataFrame(metrics)
df.columns=colNames
df.to_csv('../csv/metrics.csv')


# Create training data

tr_x=np.ones((tr.shape[0],tr.shape[1]+1))
tr_x[:,:-1]=tr
fa_x=np.zeros((fa.shape[0],fa.shape[1]+1))
fa_x[:,:-1]=fa
X=np.concatenate((tr_x,fa_x),axis=0)
np.random.shuffle(X)
y=X[:,-1]
X=X[:,:-1]

model=LogisticRegression()
model.fit(X,y)

