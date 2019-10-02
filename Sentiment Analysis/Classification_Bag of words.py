import re, math, collections, itertools, sys, os
import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures, scores
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.corpus import stopwords
import string


def evaluate_features(feature_select):
 negSentences = open(r"reviews_neg.txt", 'r', encoding='utf8')
 posSentences = open(r"reviews_pos.txt", 'r', encoding='utf8')
 negSentences = re.split(r'\n', negSentences.read())
 posSentences = re.split(r'\n', posSentences.read())

    
 posFeatures = []
 negFeatures = []
 for index,i in enumerate(posSentences):
  stop_words = set(stopwords.words('english'))
  posWords = re.findall(r"[\w']+|[.,!?;]", i)
  posWords = [w.lower() for w in posWords if not w in stop_words]
  posWords = [feature_select(posWords), 'pos']
  posFeatures.append(posWords)
 for i in negSentences:
  stop_words = set(stopwords.words('english'))
  negWords = re.findall(r"[\w']+|[.,!?;]", i)
  negWords = [w.lower() for w in negWords if not w.lower() in stop_words]
  negWords = [feature_select(negWords), 'neg']
  negFeatures.append(negWords)
        
 posCutoff = int(math.floor(len(posFeatures)*3/4))
 negCutoff = int(math.floor(len(negFeatures)*3/4))
 trainFeatures = posFeatures[:posCutoff] + negFeatures[:negCutoff]
 testFeatures = posFeatures[posCutoff:] + negFeatures[negCutoff:]
 
 classifier = NaiveBayesClassifier.train(trainFeatures)
 
 referenceSets = collections.defaultdict(set)
 testSets = collections.defaultdict(set)
 for i, (features, label) in enumerate(testFeatures): 
     referenceSets[label].add(i)               
     predicted = classifier.classify(features) 
     testSets[predicted].add(i)
 predicted_list=[]
 test_sent=open(r"reviews_test.txt", 'r', encoding='utf8')
 file_read = re.split(r'\n', test_sent.read())
 Sentences_pos = file_read[:1674]  
 Sentences_neg = file_read[1675:]  
 unseenfeat=[]
 for review in Sentences_pos: 
  stop_words = set(stopwords.words('english'))
  Words = re.findall(r"[\w']+|[.,!?;]", review)
  Words = [w.lower() for w in Words if not w.lower() in stop_words]
  #print(classifier.classify(feature_select(Words)))
  Words = [feature_select(Words), 'pos']
  unseenfeat.append(Words)
 for review in Sentences_neg: 
  stop_words = set(stopwords.words('english'))
  Words = re.findall(r"[\w']+|[.,!?;]", review)
  Words = [w.lower() for w in Words if not w.lower() in stop_words]
  Words = [feature_select(Words), 'neg']
  unseenfeat.append(Words)
 print('accuracy for unseen data:', nltk.classify.util.accuracy(classifier, unseenfeat))   
    
    
 print('train on %s instances, test on %s instances'% (len(trainFeatures), len(testFeatures)))
 print('accuracy:', nltk.classify.util.accuracy(classifier, testFeatures))
 print('pos precision:', scores.precision(referenceSets['pos'], testSets['pos']))
 print('pos recall:', scores.recall(referenceSets['pos'], testSets['pos']))
 print('neg precision:', scores.precision(referenceSets['neg'], testSets['neg']))
 print('neg recall:', scores.recall(referenceSets['neg'], testSets['neg']))
 #classifier.show_most_informative_features(10)
 


def make_full_dict(words):
 return dict([(word, True) for word in words])
 
print('using all words as features')
evaluate_features(make_full_dict)

