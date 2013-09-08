import random
import math
import matplotlib.pyplot as plt
import string
import numpy
# plot functions

plt.ion()





##########################################################
def updateValidationSet(profit, timeInMarket):
	del archiveFitness[:]
	for x in range( len( profit ) ):
		#displayArr.append(n)
		archiveFitness.append( [profit[x], timeInMarket[x]] )



def getValidationPopulation():
	trees = []

	for index in range( len(archive) ):
		trees.append( archive[index][0] )

	return trees

def randomTree(size):
	random.seed()
	treeSize = (2**size) - 1
	tree = [0]*treeSize 			# empty tree

							# top element is always a logic operation 1001 to 1006
	tree[0] = random.randint(1001,1003)
	
	for index in range( 1, treeSize ):
		parent = tree[ int(math.floor( (index-1)/2 )) ]

		if index < (2**(size-2) - 1):			# if it's not the last or second last row

			if parent < 1000:			# parent is a integer, child must be zero
				tree[index] = 0
			elif parent == 1000:			# parent is a variable, child is a range or time shift
				tree[index] = random.randint(0,45)	# bit of a cop out, max time shift == number of variables
			elif parent <= 1006:			# parent is a logical combiner, child must be another combiner or a logical operation
				tree[index] = randLogic()

			elif parent <= 1015:			# parent is a logical operation, child must be a number or operator
				tree[index] = randNumberLogic()
			
			elif parent <= 1023:			# parent is a number operator, child must be a number or operator
				tree[index] = randNumber()

		elif index < (2**(size-1) - 1):		# if it's the second last row
			if parent < 1000:			# parent is a integer, child must be zero
				tree[index] = 0
			elif parent == 1000:			# parent is a variable, child is a range or time shift
				tree[index] = random.randint(0,45)	# bit of a cop out, max time shift == number of variables
			elif parent <= 1006:			# parent is a logical combiner, Don't allow another combiner in the second last row
				tree[index] = random.randint(1010,1015)

			elif parent <= 1015:			# parent is a logical operation, child must be a number or operator
				tree[index] = randNumberLogic()
			
			elif parent <= 1023:			# parent is a number operator, child must be a number or operator
				tree[index] = randNumber()

		else:						# if it's the last row
			if parent < 1000:			# parent is a integer, child must be zero
				tree[index] = 0
			elif parent == 1000:			# parent is a variable, child is a range or time shift
				tree[index] = random.randint(0,45)	# bit of a cop out, max time shift == number of variables
			elif parent <= 1006:			# parent is a logical combiner, Don't allow another combiner in the second last row
				tree[index] = random.randint(1010,1015)	#shouldn't ever be used
				print "Invalid tree"

			elif parent <= 1015:			# parent is a logical operation, child must be a number or operator
				tree[index] = random.randint(1,999)
			
			elif parent <= 1067:			# parent is a number operator, child must be a number or operator
				tree[index] = random.randint(1,999)
			
	
	return tree

#create a random child for a logical operator
def randLogic(): 
	randNumber = random.randint(0,10);
	
	if randNumber <= 4:				# 70% chance of a logical operator
		return random.randint(1010,1015)	# return a logical operator
	else:
		return random.randint(1001,1003)	# return a logical combiner

#create a random child for a logic operator or number operation
def randNumberLogic(): 
	randNumber = random.randint(0,10);
	
	if randNumber <= 1:				# 20% chance of an integer
		return (random.random() * 1000)	# return an intger
	elif randNumber <= 4:				# 20% chance of an variable
		return 1000
	else:						# 60% change of a number operator
		return random.randint(1020,1064)	# return a number operator

#create a random child for a logic operator or number operation
def randNumber(): 
	randNumber = random.randint(0,10);
	
	if randNumber <= 3:				# 40% chance of an integer
		return (random.random() * 1000) 		# return an intger
	elif randNumber <= 8:				# 40% chance of an variable
		return 1000
	else:						# 20% change of a number operator
		return random.randint(1020,1064)	# return a number operator


def printTree(tree):
	size = int(math.log( len(tree) + 1) / math.log(2))

	lastLevel = 0;
	currentRow = []
	for index in range( len(tree) ):
		currLevel = int(math.log( index + 1) / math.log(2))
		if currLevel > lastLevel:
			offset = int( (math.pow(2,size)/4) - (currLevel / 2) )

			print('\t' * offset + '\t'.join(currentRow))
			currentRow = []
			lastLevel = currLevel
		currentRow.append(str(tree[index]))
	print('\t'.join(currentRow))
			
		
def newPopulation(trees, fitness, poolSize, treeLength):
	sortArray = []

	if len( archive ) == 0:
		archive.append([ trees[0], fitness[0] ])

	for index in range( len (fitness) ):
		dominates = False
		weakDominates = False
		forRemoval = []
		for subIndex in range( len (archive) ):
			
			if checkWeakDominance(fitness[index], archive[subIndex][1]):
				weakDominates = True

			if checkDominance(fitness[index], archive[subIndex][1]):
				dominates = True
				#print "Genome removed"
				forRemoval.append( subIndex )

		for subIndex in range( len(forRemoval) ):
			del archive[forRemoval[subIndex] - subIndex]

		if weakDominates:
			addToArchive = True
			for subIndex in range( len (archive) ):
				if checkDominance(archive[subIndex][1], fitness[index]) or checkEquals(archive[subIndex][1], fitness[index]):
					addToArchive = False
					break
			if addToArchive:
				archive.append( [trees[index], fitness[index] ] )
					

		if dominates:
			#print "New dominant genome"
			addToArchive = True
			for subIndex in range( len (archive) ):
				if checkEquals(archive[subIndex][1], fitness[index]):
					addToArchive = False
					break
			if addToArchive:
				archive.append( [trees[index], fitness[index] ] )

	print len( archive ), " dominant genomes"

	#fill the population with crossed over dominant genomes
	archive.sort(key=lambda a: a[1][0])
	displayarr = []
		
	for genome in archive:
		displayarr.append(genome[1])

	print displayarr

	newTrees = []
	for x in range(poolSize):
		mutateChance = random.randint(0, 100)
		if(x <= 500):

			newTree = mutateTree( archive[random.randint(0, len(archive)-1)][0] )
			newTrees.append( newTree )
		elif mutateChance < 10:
			newTree = randomTree(treeLength)
			newTrees.append( newTree )
		else:
			
			newTree = crossoverTrees( archive[random.randint(0, len(archive)-1)][0] , archive[random.randint(0, len(archive)-1)][0] )
			
			newTrees.append( newTree )
	

	# sort the population for graphing
	
	plotArchive()

	return newTrees

def getTreeComplexity(tree):
	freeNodes = 0

	for index in range(len(tree)):
		if(tree[index] == 0):
			freeNodes += 1

	return len(tree) - freeNodes
	

def isTreeValid(tree):
	size = int(math.log( len(tree) + 1) / math.log(2))
	start = math.pow(2, size - 1) - 1

	result = True
	for index in range(int(start), len(tree)):
		if tree[index] >= 1000:
			return False
			break
	return result
		 

def checkEquals(fitness1, fitness2):  #check if the first genome dominate the second Pareto style
	if fitness1[0] == fitness2[0] and fitness1[1] == fitness2[1]: #  and fitness1[2] <= fitness2[2]:		#if the first genomes fitness parameter isn't greater than the second then it doesn't dominate it
		return True
	else:
		return False


def checkDominance(fitness1, fitness2):  #check if the first genome dominate the second Pareto style
	if fitness1[0] > fitness2[0] and fitness1[1] < fitness2[1]  and fitness1[2] > fitness2[2]:		#if the first genomes fitness parameter isn't greater than the second then it doesn't dominate it
		return True
	else:
		return False
	
def checkWeakDominance(fitness1, fitness2):  #check if the first genome dominate the second Pareto style
	if fitness1[0] > fitness2[0] or fitness1[1] < fitness2[1] or fitness1[2] > fitness2[2]:		#if the first genomes fitness parameter isn't greater than the second then it doesn't dominate it
		return True
	else:
		return False

def mutateTree(treeR):
	tree = list(treeR)
	while True:
		index = random.randint(0, len(tree) - 1)
		value = tree[index] + random.randint(-2, 2)

		if value < 1000 and value != 0:			# parent is a integer, child must be zero
			parent = tree[ int(math.floor( (index-1)/2 )) ]
			if parent == 1000:
				tree[index] = random.randint(0, 46)
			else:
				tree[index] = value - (random.random() * 500) - 250
				if tree[index] >= 1000:
					tree[index] = 999.99
			break

	return tree


def crossoverTrees(tree1, tree2):
	crossoverNode = 0

	while True:
		crossoverNode = random.randint(1, len(tree1)-1 )
		if( sameNodeType(tree1[crossoverNode], tree2[crossoverNode]) ):
			break

	#Copy all child nodes to the first tree

	#printTree(tree1)
	#printTree(tree2)
	#print "Crossover node: ", crossoverNode
	tree1temp = tree1
	tree2temp = tree2
	
	newTree = copyChildren(tree1temp, tree2temp, crossoverNode)

	#printTree(newTree)
	return newTree


def copyChildren(tree1, tree2, offset):
	tree1[offset] = tree2[offset]

	childIndex = 2*offset + 1
	childIndex2 = 2*offset + 2

	if( childIndex >= len(tree1) or childIndex >= len(tree2)):
		return tree1
	
	tree1 = copyChildren(tree1, tree2, childIndex)
	tree1 = copyChildren(tree1, tree2, childIndex2)
	
	return tree1



def sameNodeType(node1, node2):
	if(node1 < 1000 and node2 < 1000):
		return True
	elif(node1 <= 1000 and node2 == 1000):
		return True
	elif(node1 > 1000 and node1 <= 1006 and node2 > 1000 and node2 <= 1006):
		return True
	elif(node1 >= 1010 and node1 <= 1015 and node2 >= 1010 and node2 <= 1015):
		return True
	elif(node1 >= 1020 and node1 <= 1021 and node2 >= 1020 and node2 <= 1064):
		return True
	else:
		return False


def sameStructure(tree1, tree2):
	result = True
	#print tree1, tree2
	for index in range( len(tree1) ):
		if tree1[index] < 1000 and tree2[index] < 1000:
			continue
		#elif not sameNodeType(tree1[index], tree2[index]):
		elif not tree1[index] == tree2[index]:
			result = False
			break

	return result

def degressDifferent(tree1, tree2):
	result = 0
	#print tree1, tree2
	for index in range( len(tree1) ):
		if tree1[index] < 1000 and tree2[index] < 1000:
			continue
		elif not sameNodeType(tree1[index], tree2[index]):
			result += 1
	return result

def getStructure(treeR):
	tree = list(treeR)
	outString = ""
	for index in range( len(tree) ):
		if tree[index] < 1000:
			tree[index] = 0
		outString += repr(tree[index]) + ' '

	return outString

class Pool:
	def __init__(self, db):
		self.mode = 0
		self.bestFitness = 1.0
		self.currentPool = 0
    		self.pools = []	# {archive => [tree, fitnessArr], exhausted = false}
		self.database = db['archive']

	def createPool(self, tree, fitness):
		structure = getStructure(tree)
		if not self.database.find_one({"structure": structure}):
			

			pool = {"archive": [], "staleCount": 0, "bestFitness": fitness[0]}
			pool["archive"].append([tree.tolist(), fitness])
			self.pools.append(pool)


			forDatabase = {"archive": [], "staleCount": 0, "bestFitness": fitness[0], "structure": structure}
			forDatabase["archive"].append(tree.tolist())
		
			print forDatabase
			self.database.insert(forDatabase)


			if fitness[0] > self.bestFitness:
				self.bestFitness = fitness[0]
		else:
			print 'Structure already exists, ignoring'

		return

	def processSearchMode(self, trees, fitness):
		fitnessMinimum = self.bestFitness - 0.2 * self.bestFitness

		for index in range( len (fitness) ):
			if not (fitness[index][0] > fitnessMinimum and fitness[index][0] != 0 and fitness[index][1] > 0 and fitness[index][3] > 30):
				continue
			else:
				alreadyExists = False
				for subIndex in range( len(self.pools) ):
					if sameStructure(trees[index], self.pools[subIndex]['archive'][0][0]):
						alreadyExists = True
						break
				if not alreadyExists:
					self.createPool(trees[index], fitness[index])

		notExhaustedCount = 0
		for x in self.pools:
			if x['staleCount'] < 100:
				notExhaustedCount+=1
			
		print "Pool size", len(self.pools)
		print "Not Exhausted Pools", notExhaustedCount

		if notExhaustedCount >= 10:
			self.mode = 1
			self.setOptimizePoolNumber()

	def setOptimizePoolNumber(self):
		self.pools.sort(key=lambda a: a['bestFitness'], reverse=True)

		notExhaustedCount = 0
		for x in self.pools:
			if x['staleCount'] < 100:
				notExhaustedCount+=1
		if notExhaustedCount < 5:
			self.mode = 0

		for x in range(len(self.pools)):
			if self.pools[x]['staleCount'] < 100:
				self.currentPool = x
				print "################################"
				print "Set active pool to ", x
				print "Pool fitness is ", self.pools[x]['bestFitness']
				print "################################"
				break

	def saveCurrentGenome(self):
		currentArchive = self.pools[self.currentPool]['archive']
		structure = getStructure(currentArchive[0][0])
		forDatabase = []
		for x in currentArchive:
			forDatabase.append(x[0])
		#print forDatabase
		self.database.update({"structure": structure}, {"$set": {"archive": forDatabase}})
		return


	def processOptimizeMode(self, treesR, fitnessR):
		trees = list(treesR)
		fitness = list(fitnessR)
		isStaleGeneration = True
		fitnessMinimum = self.bestFitness - 0.2 * self.bestFitness

		for index in range( len (fitness) ):

			if not (fitness[index][0] > fitnessMinimum and fitness[index][1] > 0 and fitness[index][3] > 30):
				continue

			if sameStructure(trees[index], self.pools[self.currentPool]['archive'][0][0]):
				poolLength = len(self.pools[self.currentPool]['archive'])			
				
				self.pools[self.currentPool]['archive'] = [ x for x in self.pools[self.currentPool]['archive'] if not checkDominance(fitness[index], x[1]) ]

				if len(self.pools[self.currentPool]['archive']) != poolLength:
					print "New dominant genome"
					self.pools[self.currentPool]['archive'].append([trees[index].tolist(), fitness[index] ])
					
					if fitness[index][0] > self.pools[self.currentPool]['bestFitness']:
						self.pools[self.currentPool]['bestFitness'] = fitness[index][0]

					isStaleGeneration = False
				else:
					weakDominance = True
					isDominated = False
					for x in self.pools[self.currentPool]['archive']:
						if not checkWeakDominance(fitness[index], x[1]):
							weakDominance = False
							break
							
					if weakDominance:
						print "New weak dominant genome"
						self.pools[self.currentPool]['archive'].append([trees[index].tolist(), fitness[index] ])
						isStaleGeneration = False	
				noPool = False
				break
			else:
				print "Genome didn't match the current pool. Something is messed up."

		self.pools[self.currentPool]['archive'].sort(key=lambda a: a[1][0], reverse=True)

		self.plotArchive()

		print "Stale count ", self.pools[self.currentPool]['staleCount']
		print "Pool Genomes ", len (self.pools[self.currentPool]['archive'])

		for x in self.pools[self.currentPool]['archive']:
			print x[1]

		if isStaleGeneration:
			self.pools[self.currentPool]['staleCount'] += 1
			
			if self.pools[self.currentPool]['staleCount'] > 100:
				self.saveCurrentGenome()
				self.setOptimizePoolNumber()


	def processGenomes(self, trees, fitness):
		if self.mode == 0:
			self.processSearchMode(trees, fitness)
		else:
			self.processOptimizeMode(trees, fitness)


	def newPopulation(self, poolSize, treeLength):
		newTrees = []
		for x in range(poolSize):
			if self.mode == 0:
				if len(self.pools) < 5 or x < 500:
					newTree = randomTree(treeLength)
				else:
					firstTree = self.pools[self.currentPool]['archive'][len(self.pools[self.currentPool]['archive'])-1][0]
					secondTree = self.pools[self.currentPool]['archive'][len(self.pools[self.currentPool]['archive'])-1][0]
					newTree = crossoverTrees( firstTree , secondTree )

				newTrees.append( newTree )
			else:
				firstTree = self.pools[self.currentPool]['archive'][len(self.pools[self.currentPool]['archive'])-1][0]
				secondTree = self.pools[self.currentPool]['archive'][len(self.pools[self.currentPool]['archive'])-1][0]
				newTree = crossoverTrees( firstTree , secondTree )
				newTree = mutateTree( newTree )
				newTrees.append( newTree )
		return newTrees

	def plotArchive(self):
		x = []
		y = []
		vx = []
		vy = []

		for index in range( len(self.pools) ):
			for subIndex in range(len(self.pools[index]['archive'])):
				x.append( self.pools[index]['archive'][subIndex][1][0] );
				y.append( self.pools[index]['archive'][subIndex][1][1] );
				vx.append( self.pools[index]['archive'][subIndex][1][4] );
				vy.append( self.pools[index]['archive'][subIndex][1][5] );
		#print x, y

		plt.clf()
		plt.subplot(211)
		plt.grid(color='gray', linestyle='dashed')
		#plt.axis([0, 1, 0, 100])
		plt.scatter(x, y)
		plt.subplot(212)
		plt.grid(color='gray', linestyle='dashed')
		#plt.axis([0, 1, 0, 100])
		plt.scatter(vx, vy)
		plt.draw()


	
	
