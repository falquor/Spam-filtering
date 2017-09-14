# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 13:46:15 2017

@author: QXXV1697
"""
import math
import re
import random
from sklearn.naive_bayes import MultinomialNB
import numpy as np
    
def tokenizeFromList(list_of_message):
    """return all the words from a list of message"""
    allWords = []
    for message in list_of_message:
        message = message.lower()
        messageWords = re.findall("[a-z][a-z']+", message)
        allWords += messageWords
    return list(set(allWords))

def tokenizeMessage(message):
    """return all the words from a message"""
    message = message.lower()
    words = re.findall("[a-z][a-z']+", message)
    return list(set(words))

#spamWords = tokenizeFromList(spam)
#hamWords = tokenizeFromList(ham)
    
def countWords(trainingSet):
    """ training_set is [[message,is_spam]], we want to return a dict {word:[spam_count, ham_count]}"""
    wordCount = {}
    for message,isSpam in trainingSet:
        for word in tokenizeMessage(message):
            if word not in wordCount:
                wordCount[word] = [0,0]
            if isSpam:
                wordCount[word][0] += 1
            else:
                wordCount[word][1] += 1
                
    return wordCount

#wordCount = countWords(totalSet)

def calculateProbability(wordCount, totalSpam, totalHam, k = 0.5):
    """calculate the probability for a word to be in a spam and the one to be in a ham, and put it in dict {word: [proba_spam, proba_ham]}"""
    wordProba = {}
    for word in wordCount.keys():
        wordProba[word] = [(wordCount[word][0] + k)/(totalSpam + 2*k), (wordCount[word][1] + k)/(totalHam + 2*k)]
        
    return wordProba


def spamProba(message, wordProba):
    messageWords = tokenizeMessage(message)
    logProbSpam = logProbHam = 0.0
#    probSpam = probHam = 1
    
    for word in wordProba.keys():
        
        if word in messageWords:
            logProbSpam += math.log(wordProba[word][0])
            logProbHam += math.log(wordProba[word][1])
#            probSpam = probSpam*wordProba[word][0]
#            probHam = probHam*wordProba[word][1]
            
        else:
            logProbSpam += math.log(1 - wordProba[word][0])
            logProbHam += math.log(1 - wordProba[word][1])
#            probSpam = probSpam*(1-wordProba[word][0])
#            probHam = probHam*(1-wordProba[word][1])
            
    probSpam = math.exp(logProbSpam)
    probHam = math.exp(logProbHam)
    
    print(probSpam, probHam)
    return probSpam / (probSpam + probHam)

class NaiveBayesClassifier:
    def __init__(self,k = 0.5):
        self.k = k
        self.wordProba = {}
        
    def train(self, trainingSet):
        numSpam = len([isSpam for message,isSpam in trainingSet if isSpam])
        
        numHam = len(trainingSet) - numSpam
        
        wordCount = countWords(trainingSet)
        self.wordProba = calculateProbability(wordCount, numSpam, numHam, self.k)
        
    def classify(self, message):
        return spamProba(message, self.wordProba)
    
def splitData(data, proba):
    """split data according to proba into trainingSet, testSet"""
    results = [],[]
    for row in data:
        results[0 if random.random() < proba else 1].append(row)
    return results

with open("emails/spam.txt","r", encoding = "utf-8") as file:
    spam = file.readlines()

with open("emails/ham.txt", "r", encoding = "utf-8") as file:
    ham = file.readlines()
    
totalSet = [[message,True] for message in spam] + [[message,False] for message in ham]

trainingSet, testSet = splitData(totalSet, 0.75)

classifier = NaiveBayesClassifier(1)
classifier.train(trainingSet)

classified = [(message, isSpam, classifier.classify(message)) for message, isSpam in testSet]

#training = ([tokenizeMessage(x[0]) for x in trainingSet])
#trainingValues = ([x[1] for x in trainingSet])
#
#training.reshape(-1,1)
#trainingValues.reshape(-1,1)

#clf = MultinomialNB()
#
#clf.fit(training, trainingValues)