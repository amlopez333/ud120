#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]
classifier = SVC(C = 10000, kernel = 'rbf')
#time0 = time()
classifier = classifier.fit(features_train, labels_train)
#print time() - time0
#time0 = time()
predictions = classifier.predict(features_test)
#print time() - time0
score = accuracy_score(predictions, labels_test)
print score
chris = 0
for i in predictions:
    if(i):
        chris += 1
print chris
print classifier.decision_function(features_train)   
#########################################################