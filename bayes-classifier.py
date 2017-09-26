# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 13:46:15 2017

@author: QXXV1697
"""
import math
import usefulFunctions as uf

#from sklearn.naive_bayes import MultinomialNB
#import numpy as np

#spamWords = tokenizeFromList(spam)
#hamWords = tokenizeFromList(ham)

def calculateProbability(wordCount, totalSpam, totalHam, k = 0.5):
    """calculate the probability for a word to be in a spam and the one to be in a ham, and put it in dict {word: [proba_spam, proba_ham]}"""
    wordProba = {}
    for word in wordCount.keys():
        wordProba[word] = [(wordCount[word][0] + k)/(totalSpam + 2*k), (wordCount[word][1] + k)/(totalHam + 2*k)]
        
    return wordProba


def spamProba(message, wordProba):
    """compute the 'log-likelihood ratio' that the message is a spam (spam if result > 0)"""
    messageWords = uf.tokenizeMessage(message)
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
            
#    probSpam = math.exp(logProbSpam)
#    probHam = math.exp(logProbHam)
    
    return logProbSpam - logProbHam

class NaiveBayesClassifier:
    def __init__(self,k = 0.5):
        self.k = k
        self.wordProba = {}
        
    def train(self, trainingSet):
        numSpam = len([isSpam for message,isSpam in trainingSet if isSpam])
        
        numHam = len(trainingSet) - numSpam
        
        wordCount = uf.countWords(trainingSet)
        self.wordProba = calculateProbability(wordCount, numSpam, numHam, self.k)
        
    def classify(self, message):
        return spamProba(message, self.wordProba)
    


with open("emails/spam.txt","r", encoding = "utf-8") as file:
    spam = file.readlines()

with open("emails/ham.txt", "r", encoding = "utf-8") as file:
    ham = file.readlines()
    
totalSet = [[message,True] for message in spam] + [[message,False] for message in ham]

trainingSet, testSet = uf.splitData(totalSet, 0.75)

classifier = NaiveBayesClassifier()
classifier.train(trainingSet)

classified = [(message, isSpam, classifier.classify(message)) for message, isSpam in testSet]

