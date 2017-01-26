#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import math
features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
'''
classifierArray = []
for i in range(5):
    classifierArray.append(tree.DecisionTreeClassifier(min_samples_split = 40))
weights = [1/len(classifierArray) for i in classifierArray]

def boost(features_train, labels_train):
    for i in range(len(classifierArray)):
        classifier = classifierArray[i]
        classifier.fit(features_train, labels_train)
        predictions = classifier(features_train)
        error = 1 - accuracy_score(predictions, labels_train)
        alphaWeight = 0.5*log((1-error)/error, 2)

        updateWeights(i, alphaWeight)
        boostError = getBoostTrainingError()
        if not boostError:
            break

def updateWeights(i, alphaWeight):
    weightSum = getWeightSum()
    for i in range(i, len(weights)-1):
        weights[i+1] = exp(-alphaWeight) '''

clf = AdaBoostClassifier()
clf = clf.fit(features_train, labels_train)
predictions = clf.predict(features_test)
score = accuracy_score(predictions, labels_test)
print score

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
