# -*- mode: Python; coding: utf-8 -*-

from collections import Counter, defaultdict
from classifier import Classifier
import io, sys
import numpy as np
    
def invert(dict_o):
    # map values to keys
    return {v: k for k, v in dict_o.items()}

def decipher(code, dict_o):
    decoder = invert(dict_o)
    return decoder[code]

class NaiveBaseClass:
    def calculate_relative_occurence(self, list1):
        no_examples = len(list1)
        ro_dict = dict(Counter(list1))
        for key in ro_dict.keys():
            ro_dict[key] = ro_dict[key] / float(no_examples)
        return ro_dict

    def get_max_value_key(self, d1):
        values = d1.values()
        keys = d1.keys()
        return decipher(max(values), d1)
        
    def initialize_nb_dict(self):
        self.nb_dict = {}
        for label in self.labels:
            self.nb_dict[label] = defaultdict(list)
            
class NaiveBayes(Classifier, NaiveBaseClass):
    
    def __init__(self, model={}):
        super(NaiveBayes, self).__init__(model)

    def get_model(self): return self
    def set_model(self, model): self = model
    model = property(get_model, set_model)

    def train(self, instances):
        n = len(instances)
        m = len(instances[1].features())
        Y = np.empty(n, dtype = object)
        X = np.empty((n, m), dtype = object)
        ii = 0
        for instance in instances:
            Y[ii] = instance.label
            X[ii,:] = instance.features()
            ii += 1
        self.labels = np.unique(Y)
        no_rows, no_cols = np.shape(X)
        self.initialize_nb_dict()
        self.class_probabilities = self.calculate_relative_occurence(Y)
        for label in self.labels:
            row_indices = np.where(Y == label)[0]
            X_ = X[row_indices, :]
            no_rows_, no_cols_ = np.shape(X_)
            for jj in range(0,no_cols_):
                self.nb_dict[label][jj] += list(X_[:,jj])
        for label in self.labels:
            for jj in range(0,no_cols):
                self.nb_dict[label][jj] = self.calculate_relative_occurence(self.nb_dict[label][jj])

    def classify_single_elem(self, X_elem):
        Y_dict = {}
        for label in self.labels:
            class_probability = self.class_probabilities[label]
            for ii in range(0,len(X_elem)):
              relative_feature_values = self.nb_dict[label][ii]
              if X_elem[ii] in relative_feature_values.keys():
                class_probability *= relative_feature_values[X_elem[ii]]
              else:
                class_probability *= 0
            Y_dict[label] = class_probability
        return self.get_max_value_key(Y_dict)
                    
    def classify(self, instances):
        self.predicted_Y_values = []
        n = len(instances)
        m = len(instances[1].features())
        Y = np.empty(n, dtype = object)
        X = np.empty((n, m), dtype = object)
        ii = 0
        for instance in instances:
            Y[ii] = instance.label
            X[ii,:] = instance.features()
            ii += 1
        self.labels = np.unique(Y)
        no_rows, no_cols = np.shape(X)
        for ii in range(0,no_rows):
            X_elem = X[ii,:]
            prediction = self.classify_single_elem(X_elem)
            self.predicted_Y_values.append(prediction)
        return (self.predicted_Y_values, Y, self.labels)
    
class NaiveBayesText(Classifier, NaiveBaseClass):
    """A naïve Bayes classifier."""
    
    def __init__(self, model={}):
        super(NaiveBayesText, self).__init__(model)

    def get_model(self): return self.seen
    def set_model(self, model): self.seen = model
    model = property(get_model, set_model)
    
    def initialize_nb_dict(self):
        self.nb_dict = {}
        for label in self.labels:
            self.nb_dict[label] = []

    def train(self, instances):
        Y = []
        X = []
        for instance in instances:
            Y.append(instance.label)
            X.append(instance.features())
        self.class_probabilities = self.calculate_relative_occurence(Y)
        self.labels = np.unique(Y)
        self.no_examples = len(Y)
        self.initialize_nb_dict()
        for ii in range(0,len(Y)):
            label = Y[ii]
            self.nb_dict[label] += X[ii]
        #transform the list with all occurences to a dict with relative occurences
        for label in self.labels:
            self.nb_dict[label] = self.calculate_relative_occurence(self.nb_dict[label])
                
    def classify_single_elem(self, X_elem):
        Y_dict = {}
        for label in self.labels:
            class_probability = self.class_probabilities[label]
            nb_dict_features = self.nb_dict[label]
            for word in X_elem:
                if word in nb_dict_features.keys():
                    relative_word_occurence = nb_dict_features[word]
                    class_probability *= relative_word_occurence
                else:
                    class_probability *= 0
            Y_dict[label] = class_probability
        return self.get_max_value_key(Y_dict)

    def classify(self, instances):
        self.predicted_Y_values = []
        Y = []
        X = []
        for instance in instances:
            Y.append(instance.label)
            X.append(instance.features())
        self.labels = np.unique(Y)
        n = len(X)
        for ii in range(0,n):
            X_elem = X[ii]
            prediction = self.classify_single_elem(X_elem)
            self.predicted_Y_values.append(prediction)  
        return (self.predicted_Y_values, Y, self.labels)
