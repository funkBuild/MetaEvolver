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
import math
import matplotlib.pyplot as plt

# mongo init

connection = Connection('localhost', 27017)
db = connection.genome
genomeCollection = db['genomes']
dataCollection = db['priceData']



def printFreeMemory():
	mem = pycuda.driver.mem_get_info()

	print "Free Memory:", mem[0]/1024, " KB"


def mean(winArr, lossArr):
	total = 0
	for x in range( len( winArr ) ):
		total += (winArr[x] - lossArr[x])# * (0.75 + x/100)
	
	return float( total ) / len( winArr );
	


def standardDeviation(winArr, lossArr):
	meanResult = profit(winArr, lossArr) / len(winArr)
	total = 0
	for x in range( len( winArr ) ):
		total += math.pow( (winArr[x] - lossArr[x])  - meanResult, 2)

	return math.sqrt( total / len( winArr ) )

def profit(winArr, lossArr):
	return sum(winArr) - sum(lossArr)
	
def variance(winArr, lossArr):
	meanResult = mean(winArr, lossArr)

	total = 0
	for x in range( len( winArr ) ):
		total += math.pow( math.fabs( winArr[x] - lossArr[x] - meanResult ), 2 )
	return total / len( winArr )

def profitFactor(winArr, lossArr):
	wins = 0.0
	losses = 0.0

	for x in range( len( winArr ) ):
		wins += winArr[x] 
		losses += lossArr[x]
	if(losses < 200 or wins < 500):
		return 0.0
	else:
		return wins / losses # +1 so we never divide by zero
	
def meanVariance(winArr, lossArr):
	var = variance(winArr, lossArr)
	lmean = mean(winArr, lossArr)
	if lmean <= 0:
		return 0
	else:
		return var / lmean


def expectancy(winCount, lossCount, wins, losses):
	if winCount + lossCount == 0:
		pWin = 0.0
	else:
		pWin = float(winCount) / (winCount + lossCount)
	pLoss = 1.0 - pWin

	aWin = float(wins) / winCount
	aLoss = float(losses) / lossCount

	return (pWin * aWin) - (pLoss * aLoss)

## 45 is the amount of columns in the data array
## 10 is the amount of threads in a block
mod = SourceModule("""
  
  #define TREELENGTH 63;
  #define POOLSIZE 1000;
  #define BINS 12;

  __shared__ float sharedTree[63];
  __shared__ int sharedDataLength;

  __device__ int getNodeValue(float *data, int offset, const int dataIdx);

  __device__ float getFloatNodeValue(float *data, int offset, const int dataIdx){
	int node = sharedTree[offset]; //Change to float
	
	if(offset > 30) return 0.0;
	if(node < 1000) return sharedTree[offset];

	if(node == 1000){		//variable. get value of the children and return the data value
		int child1 = getNodeValue(data, (2*offset + 1), dataIdx);			//variable
		int child2 = getNodeValue(data, (2*offset + 2), dataIdx);			//shift

		if(child1 > 45 or child1 < 0) child1 = 45;
		if(child2 > 45 or child2 < 0) child2 = 45;
	
		

		return data[(dataIdx - child2) +  sharedDataLength*child1];
	}
	else if(node <= 1064){
		float child1 = getFloatNodeValue(data, (2*offset + 1), dataIdx);
		float child2 = getFloatNodeValue(data, (2*offset + 2), dataIdx);

		float result = 0;

		switch(node){
			case 1020:
				result = child1 + child2;
				break;
			case 1021:
				result = child1 - child2;
				break;
			case 1022:
				result = fdividef(child1, child2);
				break;
			case 1023:
				result = child1 * child2;
				break;
			case 1024:
				result = acosf(child1);
				break;
			case 1025:
				result = acoshf(child1);
				break;
			case 1026:
				result = asinf(child1);
				break;
			case 1027:
				result = asinhf(child1);
				break;
			case 1028:
				result = atanf(child1);
				break;
			case 1029:
				result = atanhf(child1);
				break;
			case 1030:
				result = cbrtf(child1);
				break;
			case 1031:
				result = ceilf(child1);
				break;
			case 1032:
				result = cosf(child1);
				break;
			case 1033:
				result = coshf(child1);
				break;
			case 1034:
				result = cospif(child1);
				break;
			case 1035:
				result = erfcf(child1);
				break;
			case 1036:
				result = erfcinvf(child1);
				break;
			case 1037:
				result = erfinvf(child1);
				break;
			case 1038:
				result = erff(child1);
				break;
			case 1039:
				result = exp10f(child1);
				break;
			case 1040:
				result = exp2f(child1);
				break;
			case 1041:
				result = expf(child1);
				break;
			case 1042:
				result = expm1f(child1);
				break;
			case 1043:
				result = fabsf(child1);
				break;
			case 1044:
				result = floorf(child1);
				break;
			case 1045:
				result = lgammaf(child1);
				break;
			case 1046:
				result = log10f(child1);
				break;
			case 1047:
				result = log1pf(child1);
				break;
			case 1048:
				result = log2f(child1);
				break;
			case 1049:
				result = logbf(child1);
				break;
			case 1050:
				result = logf(child1);
				break;
			case 1051:
				result = nearbyintf(child1);
				break;
			case 1052:
				result = rcbrtf(child1);
				break;
			case 1053:
				result = rintf(child1);
				break;
			case 1054:
				result = roundf(child1);
				break;
			case 1055:
				result = rsqrtf(child1);
				break;
			case 1056:
				result = sinf(child1);
				break;
			case 1057:
				result = logf(child1);
				break;
			case 1058:
				result = sinhf(child1);
				break;
			case 1059:
				result = sinpif(child1);
				break;
			case 1060:
				result = sqrtf(child1);
				break;
			case 1061:
				result = tanf(child1);
				break;
			case 1062:
				result = tanhf(child1);
				break;
			case 1063:
				result = tgammaf(child1);
				break;
			case 1064:
				result = truncf(child1);
			
		};
		return result;
	}

	return 0.0;

  }

  __device__ int getNodeValue(float *data, int offset, const int dataIdx){
	int node = sharedTree[offset];
	
	if(offset > 30) return 0;

	if(node < 1000) return node;  	//return the integer value of the node 
	else if(node == 1000){}
	
	else if(node <= 1006){
		int child1 = getNodeValue(data, 2*offset + 1, dataIdx);
		int child2 = getNodeValue(data, 2*offset + 2, dataIdx);
		int result = 0;

		switch(node){
			case 1001:
				result = child1 && child2;
				break;
			case 1002:
				result = child1 || child2;
				break;
			case 1003:
				result = child1 ^ child2;
				break;
			case 1004:
				result = child1 && !child2;
				break;
			case 1005:
				result = child1 || !child2;
				break;
			case 1006:
				result = child1 ^ !child2;
				break;
		}
		return result;
	}
	else if(node <= 1015){
		float child1 = getFloatNodeValue(data, 2*offset + 1, dataIdx);
		float child2 = getFloatNodeValue(data, 2*offset + 2, dataIdx);
		int result = 0;

		switch(node){
			case 1010:
				result = child1 == child2;
				break;
			case 1011:
				result = child1 >= child2;
				break;
			case 1012:
				result = child1 <= child2;
				break;
			case 1013:
				result = child1 > !child2;
				break;
			case 1014:
				result = child1 < !child2;
				break;
			case 1015:
				result = child1 != !child2;
				break;
		};
		return result;
	} else if(node <= 1023) {}

	return 0;
  }

  __global__ void calc(float *tree, float *data , bool *winnerTable, int datalength)
  {
	const int treeNum = blockIdx.x;
	const int treeIndex = treeNum * TREELENGTH;
	if(threadIdx.y == 1){
		sharedDataLength = datalength;
	}

	if(threadIdx.y < 63 ){


		sharedTree[threadIdx.y] = tree[treeIndex + threadIdx.y];

	}
	__syncthreads();

	const int thredIdx = (blockIdx.y*blockDim.y + threadIdx.y);
	const int dataIdx = thredIdx;
	const int winnerOffset = datalength * treeNum;

	if(thredIdx <= 46) return;

	winnerTable[winnerOffset + thredIdx] = getNodeValue(data, 0, dataIdx);

   	return;
  }



  __global__ void fitness(float *data, bool *winnerTable, int *winCounter, int *lossCounter,int *lossTrades, int *winTrades, int *drawDown, int datalength)
  {
	const int treeNum = blockIdx.x;
	const int winnerOffset = datalength * blockIdx.x;

	const int kfolddivider = datalength / 12;
	const int counterOffset =  12 * treeNum;

	float open = -100;
	float highMark = 0;
	float balance = 0;
	float drawDownVal = 0;
	float delta = 0;

	for(int x = 1; x < datalength-1; x++){
		if(open > -99){
			delta = (open - data[x + 4*datalength]) * 10000.0;
			if(balance + delta > highMark) highMark = balance + delta;
			if(highMark - (balance + delta) > drawDownVal) drawDownVal = highMark - (balance + delta);

			if(!winnerTable[winnerOffset + x] ){
				balance += delta;
				open = -100;
	
				if(delta > 0){
					int *winCount = &winCounter[counterOffset + x / kfolddivider];
			    		atomicAdd(winCount, delta );

					int *winTrade = &winTrades[treeNum];
					atomicAdd(winTrade, 1);
				} else if(delta < 0){
					int *loseCount = &lossCounter[counterOffset + x / kfolddivider];
			    		atomicAdd(loseCount, -delta );

					int *lossTrade = &lossTrades[treeNum];
					atomicAdd(lossTrade, 1);
				}
			}
		}else if(winnerTable[winnerOffset + x] && !winnerTable[winnerOffset + x-1]){
			open = data[x + 4*datalength] - 0.0002;
		}

		

	}
	drawDown[treeNum] = drawDownVal;

  }

  __global__ void blacklistCheck(bool *winnerTable, bool *blacklistTable, int *counter, int *totalCounter)
  {
	const int treeNum = blockIdx.x;
	const int winnerOffset = datalength * blockIdx.x;
	const int blacklistOffset = datalength * threadIdx.z;
	const int thredIdx = (blockIdx.y*blockDim.y + threadIdx.y);

	winnerTable[winnerOffset + thredIdx]
	
  }


  """)


##########################
# Program start
##########################

treeLength = 6
poolSize = 1000
startDate = datetime.datetime(2010, 10, 1, 0, 0)
endDate = datetime.datetime(2011, 10, 1, 0, 0)



dataTimeSize = dataCollection.find({'datetime': {'$gt': startDate , '$lt': endDate } , 'ticker': 'AUDUSD' }).count()

data = numpy.zeros( (46,dataTimeSize) )
print "Train Data length", dataTimeSize

y = 0
mAverage = 0
aLength = 480

plot1 = []
plot2 = []
plotX = []

for dataPoint in dataCollection.find({'datetime': {'$gt': startDate , '$lt': endDate }, 'ticker': 'AUDUSD'  }).sort("datetime"):
	for x in range(len(dataPoint["data"]) ):
		data[x][y] = dataPoint["data"][x]
	if y == 0:
		mAverage = data[4][y]
	elif y > aLength:
		mAverage = mAverage - (plot1[y - aLength]/480) + (data[4][y]/480)
	plot1.append( data[4][y] )
	#data[4][y] = data[4][y] - mAverage
	plot2.append( data[4][y] )
	plotX.append(y)
	y+=1

#plt.plot(plotX, plot1)
#plt.show()
#plt.plot(plotX, plot2)
#plt.show()
print 'plot done'

#data = []
#for dataPoint in dataCollection.find({'datetime': {'$gt': startDate , '$lt': endDate }, 'ticker': 'AUDUSD' }).sort("datetime"):
#	data.append( dataPoint["data"] )


#print "#### Data length:", len(data)


#dataLength = len(data)
dataLength = dataTimeSize

pool = genome.Pool(db, 'sell', 'AUDUSD', endDate)


data = numpy.array(data).astype(numpy.float32)

printFreeMemory()
print "Data size ", data.nbytes/1024, " KB"
printFreeMemory()

data_gpu = cuda.mem_alloc(data.nbytes)
cuda.memcpy_htod(data_gpu, data)


trees = []
for x in range(poolSize):
	trees.append( genome.randomTree(treeLength) )
	
### Main Loop
generations = 0
dataDim = math.floor(dataLength/64.0)

evalArray = None
lastTrees = None
winCount = None
lossCount = None
winTradeCount = None
lossTradeCount = None
drawdownCount = None

winnerTable = numpy.zeros(dataLength * poolSize, dtype=numpy.bool)
winnerTable_gpu = cuda.mem_alloc(winnerTable.nbytes)

trees = numpy.array(trees).astype(numpy.float32)
trees_gpu = cuda.mem_alloc(trees.nbytes)

winCounter = numpy.zeros(12 * poolSize, dtype=numpy.int32)
winCounter_gpu = cuda.mem_alloc(winCounter.nbytes)

lossCounter = numpy.zeros(12 * poolSize, dtype=numpy.int32)
lossCounter_gpu = cuda.mem_alloc(lossCounter.nbytes)

winTrades = numpy.zeros(poolSize, dtype=numpy.int32)
winTrades_gpu = cuda.mem_alloc(winTrades.nbytes)

lossTrades = numpy.zeros(poolSize, dtype=numpy.int32)
lossTrades_gpu = cuda.mem_alloc(lossTrades.nbytes)

drawdown = numpy.zeros(poolSize, dtype=numpy.int32)
drawdown_gpu = cuda.mem_alloc(drawdown.nbytes)

blacklistTable_gpu = None

printFreeMemory()

checkBlacklist = False

while True:
	if len(pool.blacklist) > 0:
		checkBlacklist = True

	if checkBlacklist and pool.blacklistChanged:
		blacklist = numpy.array(pool.getBlacklist()).astype(numpy.float32)
		blacklist_gpu = cuda.mem_alloc(blacklist.nbytes)
		cuda.memcpy_htod(blacklist_gpu, blacklist)

		blacklistTable = numpy.zeros(dataLength * len(pool.blacklist), dtype=numpy.bool)
		blacklistTable_gpu = cuda.mem_alloc(blacklistTable.nbytes)

		calc(blacklist_gpu, data_gpu, blacklistTable_gpu, numpy.int32(dataLength), block=(1, 64, 1), grid=(len(pool.blacklist),int(dataDim))) # x is the tree index, y is the data index

	winnerTable = numpy.zeros(dataLength * poolSize, dtype=numpy.bool)
	cuda.memcpy_htod(winnerTable_gpu, winnerTable)

	trees = numpy.array(trees).astype(numpy.float32)
	cuda.memcpy_htod(trees_gpu, trees)

	winCounter = numpy.zeros(12 * poolSize, dtype=numpy.int32)
	cuda.memcpy_htod(winCounter_gpu, winCounter)

	lossCounter = numpy.zeros(12 * poolSize, dtype=numpy.int32)
	cuda.memcpy_htod(lossCounter_gpu, lossCounter)

	winTrades = numpy.zeros(poolSize, dtype=numpy.int32)
	cuda.memcpy_htod(winTrades_gpu, winTrades)

	lossTrades = numpy.zeros(poolSize, dtype=numpy.int32)
	cuda.memcpy_htod(lossTrades_gpu, lossTrades)

	drawdown = numpy.zeros(poolSize, dtype=numpy.int32)
	cuda.memcpy_htod(drawdown_gpu, drawdown)

	#drawdown = numpy.zeros(1000 * poolSize, dtype=numpy.float32)
	#drawdown_gpu = cuda.mem_alloc(drawdown.nbytes)
	#cuda.memcpy_htod(drawdown_gpu, drawdown)

	calc = mod.get_function("calc")
	fitness = mod.get_function("fitness")

	start = time.time()
	calc(trees_gpu, data_gpu, winnerTable_gpu, numpy.int32(dataLength), block=(1, 64, 1), grid=(poolSize,int(dataDim))) # x is the tree index, y is the data index
	fitness(data_gpu, winnerTable_gpu, winCounter_gpu, lossCounter_gpu, winTrades_gpu, lossTrades_gpu, drawdown_gpu, numpy.int32(dataLength), block=(1, 1, 1), grid=(poolSize,1)) 

	if lastTrees:
		evalArray = []

		for x in range( poolSize ):
			winArray = winCount[x*12: (x+1)*12]
			loseArray = lossCount[x*12: (x+1)*12]
			drawdownVal = drawdownCount[x]

			#stdDev = standardDeviation(winArray, loseArray)
			#prof = profit(winArray, loseArray)

			wins = winTradeCount[x]
			losses = lossTradeCount[x]
			
			wonPips = sum(winArray)
			lostPips = sum(loseArray)

			expect = expectancy(wins, losses, wonPips, lostPips)

			evalArray.append( [round(expect,3), round(wins + losses,-1), drawdownVal, wonPips, lostPips, 0] )	
 		#print evalArray
		pool.processGenomes(lastTrees, evalArray)

	lastTrees = list(trees)
	trees = pool.newPopulation(poolSize, treeLength)
		 
	winCount = numpy.empty_like(winCounter)
	cuda.memcpy_dtoh(winCount, winCounter_gpu)

	lossCount = numpy.empty_like(lossCounter)
	cuda.memcpy_dtoh(lossCount, lossCounter_gpu)

	winTradeCount = numpy.empty_like(winTrades)
	cuda.memcpy_dtoh(winTradeCount, winTrades_gpu)

	lossTradeCount = numpy.empty_like(lossTrades)
	cuda.memcpy_dtoh(lossTradeCount, lossTrades_gpu)

	drawdownCount = numpy.empty_like(drawdown)
	cuda.memcpy_dtoh(drawdownCount, drawdown_gpu)


	end = time.time()
	print "GPU Time", end - start
	
	generations += 1
	print "Generation ", generations

