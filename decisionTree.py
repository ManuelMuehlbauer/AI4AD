import numpy as np
from numpy import array
from sklearn import tree
import csv
import os
import ast

def flatten(lis):
    """Given a list, possibly nested to any level, return it flattened."""
    new_lis = []
    for item in lis:
        if type(item) == type([]):
            new_lis.extend(flatten(item))
        else:
            new_lis.append(item)
    return new_lis

def main(train_data, train_target):
	global clf
	clf = tree.DecisionTreeClassifier()
	clf.fit(train_data, train_target)

def readTraining():
	if os.path.exists("trainer.csv"):
		csv = np.genfromtxt('trainer.csv', delimiter="*", dtype=str)
		target = csv[:,1]
		data = csv[:,0]
		train_data = np.array([])
		initData = []
		for x in data:
			y = []
			# translate string to python list
			x = ast.literal_eval(x)
			#print(x)
			# pad length of array to uniformity
			arrayShapeFirst = flatten(x[0])
			# first part of representation
			if len(arrayShapeFirst) < 30:
				for i in range(len(arrayShapeFirst), 29):
					arrayShapeFirst.append(0)
			y.append(arrayShapeFirst)
			#print(arrayShapeFirst)
			# second part of representation
			arrayShapeSecond = flatten(x[1])
			if len(arrayShapeSecond) < 30:
				for i in range(len(arrayShapeSecond), 29):
					arrayShapeSecond.append(0)
			y.append(arrayShapeSecond)
			#print(arrayShapeSecond)
			# third part of representation
			arrayRules = flatten(x[2])
			for j in range(0, len(arrayRules)-1):
				arrayRules[j] = int(round(arrayRules[j]))
			if len(arrayRules) < 30:
				for i in range(len(arrayRules), 29):
					arrayRules.append(0)
			y.append(arrayRules)
			#print(arrayRules)
			# flatten complete array of representation
			x = flatten(y)
			#print(x)
			initData.append(x)
		numpyData = array(initData)
		main(numpyData, target)
	else:
		return 0

def readPrediction():
	if os.path.exists("houseInit.csv"):
		csv = np.genfromtxt('houseInit.csv', delimiter="*", dtype=str)
		data = csv
		test = []
		for x in data:
			y = []
			# translate string to python list
			x = ast.literal_eval(x)
			# pad length of array to uniformity
			arrayShapeFirst = flatten(x[0])
			# first part of representation
			if len(arrayShapeFirst) < 30:
				for i in range(len(arrayShapeFirst), 29):
					arrayShapeFirst.append(0)
			y.append(arrayShapeFirst)
			#print(arrayShapeFirst)
			# second part of representation
			arrayShapeSecond = flatten(x[1])
			if len(arrayShapeSecond) < 30:
				for i in range(len(arrayShapeSecond), 29):
					arrayShapeSecond.append(0)
			y.append(arrayShapeSecond)
			#print(arrayShapeSecond)
			# third part of representation
			arrayRules = flatten(x[2])
			for j in range(0, len(arrayRules)-1):
				arrayRules[j] = int(round(arrayRules[j]))
			if len(arrayRules) < 30:
				for i in range(len(arrayRules), 29):
					arrayRules.append(0)
			y.append(arrayRules)
			#print(arrayRules)
			# flatten complete array of representation
			x = flatten(y)
			#print(x)
			test.append(x)
		numpyData = array(test)

		predict(numpyData)

def predict(data):
	print(clf.predict(data))

readTraining()
readPrediction()