# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 22:41:55 2017

@author: Quentin
"""

import usefulFunctions as uf
from sklearn.ensemble import RandomForestClassifier
import math


with open("emails/spam.txt","r", encoding = "utf-8") as file:
    spam = file.readlines()

with open("emails/ham.txt", "r", encoding = "utf-8") as file:
    ham = file.readlines()
    
#totalSet = [[message,True] for message in spam] + [[message,False] for message in ham]

#trainingSet, testSet = uf.splitData(totalSet, 0.75)

#allWords = uf.tokenizeFromList([message for message,_ in trainingSet])

def wordVector(allWords, message):
    """transform a message into a vector of 0 and 1"""
    v = [0]*(len(allWords))
    words = uf.tokenizeMessage(message)
    for i in range(len(allWords)):
        if allWords[i] in words:
            v[i] = 1
            
    return v

#X = [wordVector(allWords, message) for message,_ in trainingSet]
#y = [ isSpam for _,isSpam in trainingSet]

#clf = RandomForestClassifier(n_estimators=10, max_features=math.floor(math.sqrt(len(allWords))))

#clf.fit(X,y)

#classified = [(isSpam,clf.predict([wordVector(allWords,message)])) for message,isSpam in testSet]

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

for isSpam, classification in classified:
    if isSpam and classification:
        true_positive += 1
    if not isSpam and classification:
        false_positive += 1
    if isSpam and not classification:
        false_negative += 1
    if not isSpam and not classification:
        true_negative += 1
        
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)

# 97% de précision et 90% de recall !!