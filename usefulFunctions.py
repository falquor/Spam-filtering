# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 22:43:25 2017

@author: Quentin
"""

import re
import random

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

def splitData(data, proba):
    """split data according to proba into trainingSet, testSet"""
    results = [],[]
    for row in data:
        results[0 if random.random() < proba else 1].append(row)
    return results