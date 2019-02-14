import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC,NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
import pickle

class VoteClassifier(ClassifierI):
    def __init__ (self,*classifier):
        self._classifiers=classifier
    def classify(self,features):
        votes=[]
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)
        return mode(votes)
    def confidence(self,features):
        votes=[]
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)
        choice=votes.count(mode(votes))
        conf=(choice/len(votes))
        return conf
pos=open("positive.txt","r").read()
neg=open("negative.txt","r").read()

documents=[]
allowed_word_types=["J"]

for r in pos.split('\n'):
    documents.append((r,"pos"))
for r in neg.split('\n'):
    documents.append((r,"neg"))

random.shuffle(documents)

#print(documents[0])
all_words=[]

word_pos=word_tokenize(pos)
word_neg=word_tokenize(neg)

for w in word_pos:
    all_words.append(w.lower())
for w in word_neg:
    all_words.append(w.lower())                   

all_words=nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:5000]

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
 
    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
#featuresets = [(find_features(rev), category) for (rev, category) in documents]
featuresets=[]
for (rev,category) in documents:
    t=()
    t=(find_features(rev),category)
    featuresets.append(t)

random.shuffle(featuresets)

training_set=featuresets[:10000]
testing_set=featuresets[10000:]

classifier=nltk.NaiveBayesClassifier.train(training_set)
print("Test set accuracy= ",(nltk.classify.accuracy(classifier,testing_set))*100.00)
#classifier.show_most_informative_features(15)

MNBClassifier=SklearnClassifier(MultinomialNB())
MNBClassifier.train(training_set)
print("MNBClassifier accuracy= ",(nltk.classify.accuracy(MNBClassifier,testing_set))*100.00)

BernoulliNBClassifier=SklearnClassifier(BernoulliNB())
BernoulliNBClassifier.train(training_set)
print("BernoulliNBClassifier accuracy= ",(nltk.classify.accuracy(BernoulliNBClassifier,testing_set))*100.00)

LogisticRegressionClassifier=SklearnClassifier(LogisticRegression())
LogisticRegressionClassifier.train(training_set)
print("LogisticRegressionClassifier accuracy= ",(nltk.classify.accuracy(LogisticRegressionClassifier,testing_set))*100.00)

SGD_Classifier=SklearnClassifier(SGDClassifier())
SGD_Classifier.train(training_set)
print("SGDClassifierClassifier accuracy= ",(nltk.classify.accuracy(SGD_Classifier,testing_set))*100.00)

SVCClassifier=SklearnClassifier(SVC())
SVCClassifier.train(training_set)
print("SVCClassifier accuracy= ",(nltk.classify.accuracy(SVCClassifier,testing_set))*100.00)

LinearSVCClassifier=SklearnClassifier(LinearSVC())
LinearSVCClassifier.train(training_set)
print("LinearSVCClassifier accuracy= ",(nltk.classify.accuracy(LinearSVCClassifier,testing_set))*100.00)

NuSVCClassifier=SklearnClassifier(NuSVC())
NuSVCClassifier.train(training_set)
print("NuSVCClassifier accuracy= ",(nltk.classify.accuracy(NuSVCClassifier,testing_set))*100.00)

vote_classifier=VoteClassifier(classifier,MNBClassifier,BernoulliNBClassifier,LogisticRegressionClassifier,SGD_Classifier,LinearSVCClassifier,LinearSVCClassifier)
print("Voted Classifier accuracy = ",(nltk.classify.accuracy(vote_classifier,testing_set))*100)
print("Classifcation = ",vote_classifier.classify(testing_set[0][0])," Confidence = ",(vote_classifier.confidence(testing_set[0][0]))*100)
print("Classifcation = ",vote_classifier.classify(testing_set[1][0])," Confidence = ",(vote_classifier.confidence(testing_set[1][0]))*100)
print("Classifcation = ",vote_classifier.classify(testing_set[2][0])," Confidence = ",(vote_classifier.confidence(testing_set[2][0]))*100)
print("Classifcation = ",vote_classifier.classify(testing_set[3][0])," Confidence = ",(vote_classifier.confidence(testing_set[3][0]))*100)
print("Classifcation = ",vote_classifier.classify(testing_set[4][0])," Confidence = ",(vote_classifier.confidence(testing_set[4][0]))*100)
print("Classifcation = ",vote_classifier.classify(testing_set[5][0])," Confidence = ",(vote_classifier.confidence(testing_set[5][0]))*100)






