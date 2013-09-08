import random
import math
import matplotlib.pyplot as plt
import string
import numpy
import time
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
	randOper = random.randint(0,70)
	randComb = random.randint(0,20)

	randNumber = random.randint(0, randOper + randComb)
	
	if randNumber <= randOper:				# 70% chance of a logical operator
		return random.randint(1010,1015)	# return a logical operator
	else:
		return random.randint(1001,1003)	# return a logical combiner

#create a random child for a logic operator or number operation
def randNumberLogic(): 
	rantInt = random.randint(0,20)
	randVar = random.randint(0,20)
	randOper = random.randint(0,70)

	randNumber = random.randint(0,rantInt + randVar + randOper);
	
	if randNumber <= rantInt:				# 20% chance of an integer
		return (random.random() * 1000)	# return an intger
	elif randNumber <= rantInt + randVar:				# 20% chance of an variable
		return 1000
	else:						# 60% change of a number operator
		return random.randint(1020,1064)	# return a number operator

#create a random child for a logic operator or number operation
def randNumber(): 
	rantInt = random.randint(0,20)
	randNum = random.randint(0,50)
	randOper = random.randint(0,20)

	randNumber = random.randint(0, rantInt + randNum + randOper);
	
	if randNumber <= rantInt:				# 40% chance of an integer
		return (random.random() * 1000) 		# return an intger
	elif randNumber <= (rantInt + randNum):				# 40% chance of an variable
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
	if fitness1[0] > fitness2[0] and fitness1[1] < fitness2[1]: #  and fitness1[2] > fitness2[2]:		#if the first genomes fitness parameter isn't greater than the second then it doesn't dominate it
		return True
	else:
		return False
	
def checkWeakDominance(fitness1, fitness2):  #check if the first genome dominate the second Pareto style
	if fitness1[0] > fitness2[0] or fitness1[1] < fitness2[1]: # or fitness1[2] > fitness2[2]:		#if the first genomes fitness parameter isn't greater than the second then it doesn't dominate it
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


#change crossover to support unequal crossover. 

def crossoverTrees(tree1R, tree2R):
	tree1 = list(tree1R)
	tree2 = list(tree2R)

	crossoverNode1 = 0
	crossoverNode2 = 0

	while True:
		crossoverNode1 = random.randint(1, len(tree1)-1 )
		crossoverNode2 = random.randint(1, len(tree1)-1 )

		if( sameNodeType(tree1[crossoverNode1], tree2[crossoverNode2]) ):
			break

	newTree = copyChildren(tree1, crossoverNode1, tree2, crossoverNode2)

	#printTree(newTree)
	return newTree


def copyChildren(tree1, offset1, tree2, offset2):
	tree1[offset1] = tree2[offset2]

	childIndex1 = 2*offset1 + 1
	childIndex2 = 2*offset1 + 2

	childIndex3 = 2*offset2 + 1
	childIndex4 = 2*offset2 + 2
	

	if( childIndex2 >= len(tree1) or childIndex3 >= len(tree2)):
		return tree1
	
	tree1 = copyChildren(tree1, childIndex1, tree2, childIndex3)
	tree1 = copyChildren(tree1, childIndex2, tree2, childIndex4)
	
	return tree1



def sameNodeType(node1, node2):
	if(node1 < 1000 and node2 < 1000 and node1 != 0 and node2 != 0):
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
	outArr = []
	for index in range( len(tree) ):
		if tree[index] < 1000:
			tree[index] = 0
		outArr.append(tree[index])

	return tuple(outArr)

class Pool:
	def __init__(self, db):
		self.bestFitness = -1000
		self.currentPool = 0
    		self.pool = []	# {archive => [tree, fitnessArr], exhausted = false}
		self.archive = []
		self.blacklist = set()
		self.database = db['archive']
		self.genomeDatabase = db['genomeArchive']
		self.staleThreshhold = 250

		#self.loadBlacklistFromDB()
		#self.loadArchiveFromDB()

	def trimArchive(self):
		self.archive.sort(key=lambda a: a[1][0], reverse=True)
		self.archive[:] = self.archive[:500]
		
	def loadBlacklistFromDB(self):
		dbStructures = self.database.find({})
		
		print "Got ", dbStructures.count(), " structures from the DB"

		for structure in dbStructures:
			self.blacklist.add(tuple(structure['structure']))

	def loadArchiveFromDB(self):
		genomes = self.genomeDatabase.find({}).sort('expectancy', -1).limit(600)
		
		for genome in genomes:
			print genome['expectancy']
			self.archive.append([genome['genome'], [ genome['expectancy'], genome['stdDeviation'], genome['trades'], genome['wins'], genome['losses']  ]])
		self.trimArchive()
		print "Got ", len(self.archive), " genomes from the DB"

	def clearArchive(self):
		self.pool = []
		self.archive = []
		self.bestFitness = -1000


	def saveCurrentGenome(self, genome):
		structure = getStructure(genome[0])
		forDatabase = []

		dbStructure = self.database.find_one({"structure": structure})
		if dbStructure == None:
			self.database.insert({"structure": structure})
			dbStructure = self.database.find_one({"structure": structure})

		structure_id = self.database.find_one({"structure": structure})['_id']

		self.genomeDatabase.insert({"parent_id": structure_id, "genome": genome[0], "expectancy": float(genome[1][0] - genome[1][1]), "trades": int(genome[1][2]), "wins": int(genome[1][3]), "losses": int(genome[1][4]) })
		return


	def processOptimizeMode(self, treesR, fitnessR):
		trees = list(treesR)
		fitness = list(fitnessR)
		isStaleGeneration = True
		fitnessMinimum = self.bestFitness - 5

		for index in range( len (fitness) ):
			if fitness[index][2] < 200 or fitness[index][0] < fitnessMinimum:
				continue
			if getStructure(trees[index].tolist()) in self.blacklist:
				#print "Structure in blacklist, skipping"
				continue

			poolLength = len(self.pool)	

			if poolLength == 0 and not (getStructure(trees[index].tolist()) in self.blacklist):
				print getStructure(trees[index].tolist())
				self.pool.append([trees[index].tolist(), fitness[index] ])
				self.archive.append([trees[index].tolist(), fitness[index] ])
				continue	
	
			toBlacklist = [ x for x in self.pool if checkDominance(fitness[index], x[1]) ]

			self.pool = [ x for x in self.pool if (not checkDominance(fitness[index], x[1])) and x[1][0] > fitnessMinimum ]

			if len(self.pool) != poolLength:
				print "New dominant genome"
				self.pool.append([trees[index].tolist(), fitness[index] ])
				self.archive.append([trees[index].tolist(), fitness[index] ])
				if fitness[index][0] > self.bestFitness:
					self.bestFitness = fitness[index][0]
			else:
				weakDominance = True
				isDominated = False
				for x in self.pool:
					if not checkWeakDominance(fitness[index], x[1]):
						weakDominance = False
						break
							
				if weakDominance:
					print "New weak dominant genome"
					self.pool.append([trees[index].tolist(), fitness[index] ])
					self.archive.append([trees[index].tolist(), fitness[index] ])


		#self.pool.sort(key=lambda a: a[1][0], reverse=True)

		# Increment the stale count
		for x in self.pool:
			#if getStructure(x[0]) in self.blacklist:
			#	print "Genome from pool is in the blacklist"
			#	print getStructure(x[0])
			x[1][5] += 1
			if x[1][5] > self.staleThreshhold: #if greater than the threshhold
				print "Adding structure to blacklist"
				print getStructure(x[0])
				self.blacklist.add( getStructure(x[0]) )
				self.saveCurrentGenome(x)
				
		self.trimArchive()
		#remove any expired genomes
		self.pool = [ x for x in self.pool if not x[1][5] > self.staleThreshhold]
				

		self.plotArchive()

		print "Pool Genomes ", len (self.archive)


		for x in self.pool:
			print x[1]


	def processGenomes(self, trees, fitness):
		self.processOptimizeMode(trees, fitness)


	def newPopulation(self, poolSize, treeLength):
		currTime = time.time()
		millis = (currTime - int(currTime)) * 100000
		random.seed( millis )

		newTrees = []
		for x in range(poolSize):
			if len(self.pool) == 0 or x < 100:
				newTrees.append( randomTree(treeLength) )	
			elif x < 350:
				firstTree = self.archive[ random.randint(0, len(self.archive)-1 ) ][0]
				secondTree = randomTree(treeLength)
				newTree = crossoverTrees( firstTree , secondTree )
				newTrees.append( newTree )
			else:
				firstTree = self.archive[ random.randint(0, len(self.archive)-1 ) ][0]
				secondTree = self.archive[ random.randint(0, len(self.archive)-1 ) ][0]
				newTree = crossoverTrees( firstTree , secondTree )
				newTree = mutateTree( newTree )
				newTrees.append( newTree )
		return newTrees

	def plotArchive(self):
		x = []
		y = []

		for index in range( len(self.pool) ):
			for subIndex in range(len(self.pool)):
				x.append( self.pool[subIndex][1][0] );
				y.append( self.pool[subIndex][1][1] );

		plt.clf()
		plt.grid(color='gray', linestyle='dashed')
		plt.scatter(x, y)
		plt.draw()


	
	
