import re, math, collections, itertools, sys, os
import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures, scores
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.corpus import stopwords
import string
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.metrics import accuracy_score

negSentences = open(r"reviews_neg.txt", 'r', encoding='utf8')
posSentences = open(r"reviews_pos.txt", 'r', encoding='utf8')
testSentences = open(r"reviews_test.txt", 'r', encoding='utf8')
negSentences = re.split(r'\n', negSentences.read())

posSentences = re.split(r'\n', posSentences.read())
testSentences = re.split(r'\n', testSentences.read())


corpus=[]
labels=[]
test=[]
labels_test=[]

for line in posSentences:
  stop_words = set(stopwords.words('english'))
  posWords = re.findall(r"[\w']+|[.,!?;]", line)
  posWords = [w.lower() for w in posWords if not w in stop_words]
  text=" ".join(posWords)
  corpus.append(text)
  labels.append("pos")

for line in negSentences:
  stop_words = set(stopwords.words('english'))
  posWords = re.findall(r"[\w']+|[.,!?;]", line)
  posWords = [w.lower() for w in posWords if not w in stop_words]
  text=" ".join(posWords)
  corpus.append(text)
  labels.append("neg")

for i,line in enumerate(testSentences):
  stop_words = set(stopwords.words('english'))
  posWords = re.findall(r"[\w']+|[.,!?;]", line)
  posWords = [w.lower() for w in posWords if not w in stop_words]
  text=" ".join(posWords)
  test.append(text)
  if i<1675:
   labels_test.append("pos")
  else:
   labels_test.append("neg")


stop_words = set(stopwords.words('english'))
posWords = re.findall(r"[\w']+|[.,!?;]", "Complete waste of time. Not even a single star from me")
posWords = [w.lower() for w in posWords if not w in stop_words]
text=" ".join(posWords)
test.append(text)
labels_test.append("neg")



count_vect=CountVectorizer()
tfidf_transformer=TfidfTransformer()
counts=count_vect.fit_transform(corpus)
Z=tfidf_transformer.fit_transform(counts)

predict = count_vect.transform(test)
unclas=tfidf_transformer.transform(predict)

clf = MultinomialNB()
clf.fit(Z,labels)

predicted_bayes=clf.predict(unclas)

#print(predicted_bayes)
print(accuracy_score(predicted_bayes, labels_test))