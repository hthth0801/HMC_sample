# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 19:11:26 2015
load several dataset for the bayesian logistic regression model
From UAI
@author: tian
"""

import numpy as np
import urllib
from theano import config

def load_german_credit():
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric"
    raw_data = urllib.urlopen(url)
    german_credit = np.genfromtxt(raw_data)
    data = german_credit[:,:-1]
    label = german_credit[:, -1:]
    label[label==2.] = 0.
    return data.astype(config.floatX), label.flatten().astype(config.floatX)
    
def load_australian_credit():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat"
    raw_data = urllib.urlopen(url)
    german_credit = np.genfromtxt(raw_data)
    data = german_credit[:,:-1]
    label = german_credit[:, -1:]
    #label[label==0.] = -1.
    return data.astype(config.floatX), label.flatten().astype(config.floatX)
    
def load_pima_indian():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
    raw_data = urllib.urlopen(url)
    indian = np.loadtxt(raw_data, delimiter=",")
    data = indian[:, :-1]
    label = indian[:, -1:]
    #label[label==0.] = -1.
    return data.astype(config.floatX), label.flatten().astype(config.floatX)
    
def load_heart():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat'
    raw_data = urllib.urlopen(url)
    heart = np.genfromtxt(raw_data)
    data = heart[:,:-1]
    label = heart[:, -1:]
    label[label==2.] = 0. # presence of the heart disease
    return data.astype(config.floatX), label.flatten().astype(config.floatX)

    