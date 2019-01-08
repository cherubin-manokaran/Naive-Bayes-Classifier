# -*- mode: Python; coding: utf-8 -*-

from __future__ import division

from corpus import Document, BlogsCorpus, NamesCorpus
from naive_bayes_improved_2 import NaiveBayesText
from collections import Counter, defaultdict

import sys
from random import shuffle, seed
from unittest import TestCase, main, skip

import numpy as np

class EvenOdd(Document):
    def features(self):
        """Is the data even or odd?"""
        return [self.data % 2 == 0]

class BagOfWords(Document):
    def features(self):
        """Trivially tokenized words."""
        return self.data.split()

class Name(Document):
    def features(self, letters="abcdefghijklmnopqrstuvwxyz"):
        """From NLTK's names_demo_features: first & last letters, how many
        of each letter, and which letters appear."""
        name = self.data
        return ([name[0].lower(), name[-1].lower()] +
                [name.lower().count(letter) for letter in letters] +
                [letter in name.lower() for letter in letters])

def true_positives(determined_Y, real_Y, label):
    true_positives = 0
    for ii in range(0,len(determined_Y)):
        if determined_Y[ii] == label and real_Y[ii] == label: 
            true_positives += 1
    return true_positives

def all_positives(determined_Y, label):
    return Counter(determined_Y)[label]

def false_negatives(determined_Y, real_Y, label):
    false_negatives = 0
    for ii in range(0,len(determined_Y)):
        if determined_Y[ii] != label and real_Y[ii] == label: 
            false_negatives += 1
    return false_negatives
                  
def precision(determined_Y, real_Y, label):
    if float(all_positives(determined_Y, label)) == 0: return 0
    return true_positives(determined_Y, real_Y, label) / float(all_positives(determined_Y, label))

def recall(determined_Y, real_Y, label):
    denominator = float((true_positives(determined_Y, real_Y, label) + false_negatives(determined_Y, real_Y, label)))
    if denominator == 0: return 0
    return true_positives(determined_Y, real_Y, label) / denominator

def f1_score(determined_Y, real_Y, label):
    p = precision(determined_Y, real_Y, label)
    r = recall(determined_Y, real_Y, label)
    if p + r == 0: return 0
    f1 = 2 * (p * r) / (p + r)
    return f1

def accuracy(classifier, test, verbose=sys.stderr):
    predicted_Y, Y_test, unique_labels = classifier.classify(test)
    correct = np.empty(len(predicted_Y), dtype = object)
    for ii in range(0,len(predicted_Y)):
        correct[ii] = predicted_Y[ii] == test[ii].label    
    if verbose:
        print("%.2i%% " % (100 * np.sum(correct) / len(correct)), file=verbose)
    for y_label in unique_labels:
        f1 = f1_score(predicted_Y, Y_test, y_label)
        print("F1-score on the test-set for class %s is: %s" % (y_label, f1))
    return np.sum(correct) / len(correct)

class NaiveBayesTest(TestCase):
    u"""Tests for the naïve Bayes classifier."""

    def split_blogs_corpus(self, document_class):
        """Split the blog post corpus into training and test sets"""
        blogs = BlogsCorpus(document_class=document_class)
        self.assertEqual(len(blogs), 3232)
        seed(hash("blogs"))
        shuffle(blogs)
        return (blogs[:3000], blogs[3000:])

    def test_blogs_bag(self):
        """Classify blog authors using bag-of-words"""
        train, test = self.split_blogs_corpus(BagOfWords)
        classifier = NaiveBayesText()
        classifier.train(train)
        self.assertGreater(accuracy(classifier, test), 0.55)

    def split_blogs_corpus_imba(self, document_class):
        blogs = BlogsCorpus(document_class=document_class)
        imba_blogs = blogs.split_imbalance()
        return (imba_blogs[:1600], imba_blogs[1600:])

    def test_blogs_imba(self):
        train, test = self.split_blogs_corpus_imba(BagOfWords)
        classifier = NaiveBayesText()
        classifier.train(train)
        # you don't need to pass this test
        self.assertGreater(accuracy(classifier, test), 0.1)
        
if __name__ == '__main__':
    # Run all of the tests, print the results, and exit.
    main(verbosity=2)
