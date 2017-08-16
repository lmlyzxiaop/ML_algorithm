from numpy import *
import math
import matplotlib.pyplot as plt


def loadDataSet(filename):
	dataMat = []
	labelMat = []
	fr = open(filename)
	for line in fr.readlines():
		lineArr = line.strip('\n').split('\t')
		dataMat.append([1,float(lineArr[0]),float(lineArr[1])])
		labelMat.append(int(lineArr[2]))
	return dataMat,labelMat

def sigmoid(inX):
	return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn,classLabels):
	dataMatrix = mat(dataMatIn)
	labelMat = mat(classLabels).transpose()
	m,n = shape(dataMatrix)
	alpha = 0.001
	maxCycles = 1000
	weights = ones((n,1))
	for k in range(maxCycles):
		h = sigmoid(dataMatrix*weights)
		error = labelMat - h
		weights += alpha*dataMatrix.transpose()*error
	return weights

def stocGradAscent(dataMatIn,classLabels):
	m,n = shape(dataMatIn)
	alpha = 0.01
	weights = ones(n)
	for i in range(m):
		h = sigmoid(sum(dataMatIn[i]*weights))
		error = classLabels[i] - h
		weights += alpha*error*dataMatIn[i]
	return weights


def stocGradAscent1(dataMatIn,classLabels,numIter=150):
	m,n = shape(dataMatIn)
	weights = ones(n)
	count  = []
	for j in range(numIter):
		dataIndex = range(m)
		for i in range(m):
			alpha = 4/(1.0+i+j) + 0.0001
			randIndex = int(random.uniform(0,len(dataIndex)))
			h = sigmoid(sum(dataMatIn[randIndex]*weights))
			error = classLabels[randIndex] - h
			weights += alpha*error*dataMatIn[randIndex]
			count.append(weights[0])
			del(dataIndex[randIndex])
	return weights,count

def plotBestFit(weights,filename):
	#weights = wei.getA()
	dataMat,labelMat = loadDataSet(filename)
	dataArr = array(dataMat)
	n = shape(dataArr)[0]
	xcord1 = [];ycord1 = []
	xcord2 = [];ycord2 = []
	for i in range(n):
		if int(labelMat[i]) == 1:
			xcord1.append(dataArr[i,1])
			ycord1.append(dataArr[i,2])
		else:

			xcord2.append(dataArr[i,1])
			ycord2.append(dataArr[i,2])

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
	ax.scatter(xcord2,ycord2,s=20,c='green')
	x = arange(-3.0,3.0,0.1)
	y = (-weights[0]-weights[1]*x)/weights[2]
	ax.plot(x,y)
	plt.xlabel('X1')
	plt.ylabel('X2')
	plt.show()

def plotXi(count):
	n = shape(count)[0]
	fig = plt.figure()
	ax = fig.add_subplot(111)
	#ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
	#ax.scatter(xcord2,ycord2,s=20,c='green')
	x = arange(0,150000,10)
	y = count
	ax.plot(x,y)
	plt.xlabel('epoch')
	plt.ylabel('X2')
	plt.show()

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0


def colicTest():
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights,cnt = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print "the error rate of this test is: %f" % errorRate
    return errorRate

def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print "after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests))


if __name__ == '__main__':
		
	filename = 'testSet.txt'
	dataArr,labelMat = loadDataSet(filename)
	weights,count= stocGradAscent1(array(dataArr),labelMat)
	plotBestFit(weights,filename)
	plotXi(count)
	multiTest()



