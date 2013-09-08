import pycuda.driver as cuda
#import readCSV
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit
import numpy
import time
from pycuda.compiler import SourceModule
import metaGenome as genome
from pymongo import Connection
import datetime

def getFloatNodeValue(tree, offset, data, dataoffset):
	if(offset > len(tree) ):
		print "Called beyond tree, something fucked up"
	print "Node: ", tree[offset], "Children : ", 2*offset+1,2*offset+2 
	node = tree[offset]
	if(node < 1000):
		return node
	if(node == 1000):
		child1 = int(getNodeValue(tree, 2*offset + 1, data, dataoffset))
		child2 = int(getNodeValue(tree, 2*offset + 2, data, dataoffset))
		#print "Get var and offset", child1, child2
		dataValue = 0
		try:
			if((dataoffset - child2) <= 0 or child2 < 0):
				#print "Variable out of range"
				dataValue = 0
			else:
				if(child1 >= 45 or child1 <= 0):
					dataValue = data[dataoffset - child2][0]
				else:
					dataValue = data[dataoffset - child2][child1]
		except:
			print "Get var and offset", child1, child2

		return dataValue
	if(node <= 1023):
		child1 = getFloatNodeValue(tree, 2*offset + 1, data, dataoffset)
		child2 = getFloatNodeValue(tree, 2*offset + 2, data, dataoffset)
	
		if(node == 1020):
			return (child1 + child2)
		elif(node == 1021):
			return (child1 - child2)
		elif(node == 1022):
			if(child2 == 0):
				return 0
			else:
				return (child1 / child2)
		elif(node == 1023):
			return (child1 * child2)

	print "Fall Through! something is fucked"

def getNodeValue(tree, offset, data, dataoffset):
	if(offset > len(tree) ):
		print "Called beyond tree, something fucked up"
	print "Node: ", tree[offset], "Children : ", 2*offset+1,2*offset+2 
	node = tree[offset]
	

	if(node < 1000):
		#print "Integer: ", node
		return node

	

	if(node <= 1006):
		child1 = getNodeValue(tree, 2*offset + 1, data, dataoffset)
		child2 = getNodeValue(tree, 2*offset + 2, data, dataoffset)
		
		#print "Logical Operator on : ", child1, child2
	
		if(node == 1001):
			#print (child1 and child2)
			return (child1 and child2)
		elif(node == 1002):
			#print (child1 or child2)
			return (child1 or child2)
		elif(node == 1003):
			#print (child1 != child2)
			return (child1 != child2)

		#elif(node == 1004):
			#return (child1 or not child2)
		#elif(node == 1005):
			#return (child1 and not child2)
		#elif(node == 1006):
			#return (child1 != not child2)
	if(node <= 1015):
		child1 = getFloatNodeValue(tree, 2*offset + 1, data, dataoffset)
		child2 = getFloatNodeValue(tree, 2*offset + 2, data, dataoffset)

		
	
		if(node == 1010):
			#print child1, "==", child2, "=", child1 == child2
			return (child1 == child2)
		elif(node == 1011):
			#print child1, ">=", child2, "=", child1 >= child2
			return (child1 >= child2)
		elif(node == 1012):
			#print child1, "<=", child2, "=", child1 <= child2
			return (child1 <= child2)
		elif(node == 1013):
			#print child1, ">", child2, "=", child1 > child2
			return (child1 > child2)
		elif(node == 1014):
			#print child1, "<", child2, "=", child1 < child2
			return (child1 < child2)
		elif(node == 1015):
			#print child1, "!=", child2, "=", child1 != child2
			return (child1 != child2)
	print "Fall Through! something is fucked"

# winner table functino

def makeWinnerTable(data, stoploss):
	spread = 2.0
	winnerTable = []
	
	for step in range( len(data) ):
		result = False
		open = data[step][5] - (spread / 10000.0)
		
		for x in range(30):
			if (step + x) >= len(data):
				break
			delta = (data[step+x][2] - open) * 10000.0
			if delta >= stoploss:
				result = True
				break
			elif delta <= (-1 * stoploss):
				break
		winnerTable.append(result)
	return winnerTable

def getMaxWinnerValue(winnerTable):
	trueCount = 0
	falseCount = 0

	for value in winnerTable:
		if(value):
			trueCount += 1
		else:
			falseCount += 1

	return [trueCount, falseCount]

# mongo init

connection = Connection('localhost', 27017)
db = connection.genome
genomeCollection = db['genomes']
dataCollection = db['priceData']


##########################
# Program start
##########################

treeLength = 5
poolSize = 100

data = []

for dataPoint in dataCollection.find({'datetime': {'$gt': datetime.datetime(2011, 4, 18, 0, 0) , '$lt': datetime.datetime(2011, 4, 19, 0, 0) }  }).sort("datetime"):
	data.append( dataPoint["data"] )


winnerTable = makeWinnerTable(data, 20)
maxWinnerValue = getMaxWinnerValue(winnerTable)
#print maxWinnerValue


#### transfer data array and winner table to GPU
while True:

	trees = []
	for x in range(poolSize):
		trees.append( genome.randomTree(treeLength) )
	
	### Main Loop
	generations = 0
	noChangeCount = 0
	lastBestFitness = -99999999999999

	while True:
		winCounter = [0]*len(trees)
		lossCounter = [0]*len(trees)
		fitness = [0]*len(trees)

		for treeIndex in range(len(trees)):
			print trees[treeIndex]
			tree = trees[treeIndex]

			for dataIndex in range(len(data)):
				result = getNodeValue(tree, 0, data, dataIndex)
				#print result

				if(result and winnerTable[dataIndex]):
					winCounter[treeIndex] += 1
				elif(result and not winnerTable[dataIndex]):
					lossCounter[treeIndex] += 1
		

		evalArray = []
		displayArr = []

		for x in range( len( winCounter ) ):
			if(lossCounter[x] + winCounter[x] == 0):
				p = 0
			else:
				p = round(winCounter[x] / float(lossCounter[x] + winCounter[x]), 3)

			if(winCounter[x] == maxWinnerValue[0] and lossCounter[x] == maxWinnerValue[1] ):	#eliminate the always on case
				n = 0
				p = 0
			else:
				n = winCounter[x]

			displayArr.append(n)
			evalArray.append( [p, n] )
 
		print "Current P: ", displayArr
		generations += 1

		print "Generation ", generations

		trees = genome.newPopulation(trees, evalArray, poolSize, treeLength)
		
	break
		

print ""
print "GPU Time :", gpuTime
print "GPU Result: ", fitness




