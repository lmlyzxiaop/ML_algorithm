from numpy import *
import numpy as np
import operator
import matplotlib.pyplot as plt
from array import array

def classify0(inX,dataSet,labels,k):
	dataSetSize = dataSet.shape[0] #only read the row of the matrix
	diffMat = tile(inX,(dataSetSize,1))-dataSet
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances**0.5
	sortedDistIndicies = distances.argsort()
	classCount = {}
	for i in range(k):
		voteIlabel =labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 # the 0 is a default return value
		sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
	return sortedClassCount[0][0]

def file2matrix(filename):
	fr = open(filename)
	arrayOlines = fr.readlines()
	numberOfLines = len(arrayOlines)
	returnMat = np.zeros((numberOfLines,3))
	classLabelVector = []
	index = 0
	for line in arrayOlines:
		line = line.strip() #delete the '\n'
		listFromLine = line.split('\t') #delete the '\n'
		returnMat[index,:] = listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1])) # append the labels
		index+=1
	return returnMat,classLabelVector

def display(filename):
	dataMat,dataLabels = file2matrix(filename)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(dataMat[:,1],dataMat[:,2],15.0*np.array(dataLabels),15.0*np.array(dataLabels))
	#ax.scatter(dataMat[:,1],dataMat[:,2])
	plt.show()

def autoNorm(dataSet):
	minVals = dataSet.min(0) #the 0 means choose the minval from columns
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = np.zeros(np.shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet - tile(minVals,(m,1))
	normDataSet = normDataSet/tile(ranges,(m,1))
	return normDataSet,ranges,minVals

def datingClassTest(filename):
    hoRatio = 0.2      #hold out 10%
    datingDataMat,datingLabels = file2matrix(filename)       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %d, the real answer is: %d" %(classifierResult, datingLabels[i])
        #print(classifierResult)
        #print(datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total acurracy rate is: %f" % (1 - errorCount/float(numTestVecs))
    print errorCount

if __name__=='__main__':
	filepath = 'datingTestSet2.txt'
	
	datingMat,datingLabel = file2matrix(filepath)
	datingClassTest(filepath)
	display(filepath)
	




