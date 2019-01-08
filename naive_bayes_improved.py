# -*- mode: Python; coding: utf-8 -*-

from collections import Counter, defaultdict
from classifier import Classifier
import io, sys
import numpy as np
import math as calc
import re, string
import csv

def invert(dict_o):
    return {v: k for k, v in dict_o.items()}

def decipher(code, dict_o):
    decoder = invert(dict_o)
    return decoder[code]

def remove_punctuation(text):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    return regex.sub('', text)

def compile_stop_words():
    stopset = []
    with open('stopwords.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            stopset.append(row)     
    return stopset
            
def tokenize(text):
    text = remove_punctuation(text)
    text = text.lower()
    text = re.split("\W+", text)
    words = []
    stopset = compile_stop_words()
    for word in text:
        if word not in stopset[0]:
            words.append(word)
    return words

class NaiveBaseClass:
    def calculate_relative_class_occurence(self, list1):
        no_examples = len(list1)
        ro_dict = dict(Counter(list1))
        for key in ro_dict.keys():
            ro_dict[key] = ro_dict[key] / float(no_examples)
        return ro_dict
    
    def calculate_relative_occurence(self, list1):
        no_examples = len(list1)
        ro_dict = dict(Counter(list1))
        co_dict = ro_dict
        for key in ro_dict.keys():
            ro_dict[key] = ro_dict[key] / float(no_examples)
        return [no_examples, co_dict, ro_dict]

    def get_max_value_key(self, d1):
        values = d1.values()
        keys = d1.keys()
        return decipher(max(values), d1)
        
    def initialize_nb_dict(self):
        self.nb_dict = {}
        for label in self.labels:
            self.nb_dict[label] = defaultdict(list)
            
class NaiveBayesText(Classifier, NaiveBaseClass):
    """A naïve Bayes classifier."""
    
    def __init__(self, model={}):
        super(NaiveBayesText, self).__init__(model)

    def get_model(self): return self
    def set_model(self, model): self = model
    model = property(get_model, set_model)   
    
    def initialize_nb_dict(self):
        self.nb_dict = {}
        for label in self.labels:
            self.nb_dict[label] = []

    def train(self, instances):
        Y = []
        X = []
        self.no_word_examples = {}
        self.co_dict = {}
        for instance in instances:
            Y.append(instance.label)
##            X.append(instance.features())
            text = " ".join(instance.features())
            features = tokenize(text)
            X.append(features)
        self.class_probabilities = self.calculate_relative_class_occurence(Y)
        self.labels = np.unique(Y)
        self.no_examples = len(Y)
        self.initialize_nb_dict()
        for ii in range(0,len(Y)):
            label = Y[ii]
            self.nb_dict[label] += X[ii]
        #transform the list with all occurences to a dict with relative occurences
        for label in self.labels:
            self.no_word_examples[label], self.co_dict[label], self.nb_dict[label] = self.calculate_relative_occurence(self.nb_dict[label])
                
    def classify_single_elem(self, X_elem):
        Y_dict = {}
        for label in self.labels:
            class_probability = self.class_probabilities[label]
            nb_dict_features = self.nb_dict[label]
            for word in X_elem:
                if word in nb_dict_features.keys():
                    relative_word_occurence = nb_dict_features[word]
                    p = np.mean([(self.co_dict[label][word] + 1) / (len(self.co_dict[label]) + self.no_word_examples[label]), relative_word_occurence])
                    class_probability *= p
                else:
                    relative_word_occurence = 0
                    p = np.mean([1 / (len(self.co_dict[label]) + self.no_word_examples[label]), relative_word_occurence])
                    class_probability *= p
            Y_dict[label] = class_probability
        return self.get_max_value_key(Y_dict)

    def classify(self, instances):
        self.predicted_Y_values = []
        Y = []
        X = []
        for instance in instances:
            Y.append(instance.label)
##            X.append(instance.features())
            text = " ".join(instance.features())
            features = tokenize(text)
            X.append(features)
        self.labels = np.unique(Y)
        n = len(X)
        for ii in range(0,n):
            X_elem = X[ii]
            prediction = self.classify_single_elem(X_elem)
            self.predicted_Y_values.append(prediction)  
        return (self.predicted_Y_values, Y, self.labels)
