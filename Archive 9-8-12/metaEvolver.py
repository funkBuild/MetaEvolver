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

# mongo init

connection = Connection('localhost', 27017)
db = connection.genome
genomeCollection = db['genomes']
dataCollection = db['priceData']


def mean(winArr, lossArr):
	total = 0
	for x in range( len( winArr ) ):
		total += (winArr[x] - lossArr[x])# * (0.75 + x/100)
	
	return float( total ) / len( winArr );
	


def standardDeviation(winArr, lossArr):
	#meanResult = mean(winArr, lossArr)
	meanResult = sum(lossArr) / len(lossArr)
	total = 0
	for x in range( len( winArr ) ):
		total += math.pow( lossArr[x]  - meanResult, 2)

	return math.sqrt( total / len( lossArr ) )

def profit(winArr, lossArr):
	total = 0
	for x in range( len( winArr ) ):
		total += winArr[x] - lossArr[x]
	return total
	
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


## 45 is the amount of columns in the data array
## 10 is the amount of threads in a block
mod = SourceModule("""
  
  #define TREELENGTH 63;
  #define POOLSIZE 1000;
  #define BINS 12;

  __shared__ float sharedTree[63];

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

		return data[(dataIdx - 46 * child2)+  child1];
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


	if(threadIdx.y < 63 ){


		sharedTree[threadIdx.y] = tree[treeIndex + threadIdx.y];

	}
	__syncthreads();

	const int thredIdx = (blockIdx.y*blockDim.y + threadIdx.y);
	const int dataIdx = 46*thredIdx;
	const int winnerOffset = datalength * treeNum;

	if(thredIdx <= 46) return;

	winnerTable[winnerOffset + thredIdx] = getNodeValue(data, 0, dataIdx);

   	return;
  }



  __global__ void fitness(float *data, bool *winnerTable, int *winCounter, int *lossCounter,int *timeInMarket, float *drawdown, int datalength)
  {
	const int thredIdx = (blockIdx.y*blockDim.y + threadIdx.y);
	const int dataIdx = 46*thredIdx;
	const int treeNum = blockIdx.x;
	const int winnerOffset = datalength * blockIdx.x;

	const int kfolddivider = datalength / 12;
	const int counterOffset =  12 * treeNum;

	if(winnerTable[winnerOffset + thredIdx] && !winnerTable[winnerOffset + thredIdx -1]){
		float open = data[dataIdx + 4] + 0.0002;
		float thisDrawDown = 0;
		//int marketCounter = 0;
		

		for(int x = thredIdx; x < datalength-1; x++){
			//marketCounter++;

			//float delta = (data[46*x + 4] - open ) * 10000.0;
			//if(thisDrawDown > delta) thisDrawDown = delta;


			if(!winnerTable[winnerOffset + x] || x >= datalength-2){
				float delta = (data[46*x + 4] - open ) * 10000.0;
				if(delta > 0){
					int *winCount = &winCounter[counterOffset + x / kfolddivider];
		    			atomicAdd(winCount, delta );
				} else {

					int *loseCount = &lossCounter[counterOffset + x / kfolddivider];
		    			atomicAdd(loseCount, -delta );
				}

				int *marketCount = &timeInMarket[treeNum];
				atomicAdd(marketCount, 1);
				//drawdown[blockIdx.y + 1000 * treeNum] = thisDrawDown;
				return;
			}
		}
		
	}

  }
  """)


##########################
# Program start
##########################

treeLength = 6
poolSize = 1000
startDate = datetime.datetime(2011, 6, 1, 0, 0)
endDate = datetime.datetime(2011, 10, 1, 0, 0)
endValidationDate = datetime.datetime(2011, 11, 1, 0, 0)

data = []
validData = []

for dataPoint in dataCollection.find({'datetime': {'$gt': startDate , '$lt': endDate }, 'ticker': 'AUDUSD' }).sort("datetime"):
	data.append( dataPoint["data"] )

for dataPoint in dataCollection.find({'datetime': {'$gt': endDate , '$lt': endValidationDate }, 'ticker': 'AUDUSD' }).sort("datetime"):
	validData.append( dataPoint["data"] )


print "#### Data length:", len(data)
print "#### Valid data length:", len(validData)

dataLength = len(data)
validDataLength = len(validData)

pool = genome.Pool(db)


data = numpy.array(data).astype(numpy.float32)
data_gpu = cuda.mem_alloc(data.nbytes)
cuda.memcpy_htod(data_gpu, data)

validData = numpy.array(validData).astype(numpy.float32)
validData_gpu = cuda.mem_alloc(validData.nbytes)
cuda.memcpy_htod(validData_gpu, validData)

#### transfer data array and winner table to GPU
while True:

	trees = []
	for x in range(poolSize):
		trees.append( genome.randomTree(treeLength) )
	
	### Main Loop
	generations = 0

	while True:
	

		winnerTable = numpy.zeros(len(data) * poolSize, dtype=numpy.bool)
		winnerTable_gpu = cuda.mem_alloc(winnerTable.nbytes)
		cuda.memcpy_htod(winnerTable_gpu, winnerTable)

		trees = numpy.array(trees).astype(numpy.float32)
		trees_gpu = cuda.mem_alloc(trees.nbytes)
		cuda.memcpy_htod(trees_gpu, trees)

		winCounter = numpy.zeros(12 * poolSize, dtype=numpy.int32)
		winCounter_gpu = cuda.mem_alloc(winCounter.nbytes)
		cuda.memcpy_htod(winCounter_gpu, winCounter)

		lossCounter = numpy.zeros(12 * poolSize, dtype=numpy.int32)
		lossCounter_gpu = cuda.mem_alloc(lossCounter.nbytes)
		cuda.memcpy_htod(lossCounter_gpu, lossCounter)

		timeInMarket = numpy.zeros(poolSize, dtype=numpy.int32)
		timeInMarket_gpu = cuda.mem_alloc(timeInMarket.nbytes)
		cuda.memcpy_htod(timeInMarket_gpu, timeInMarket)

		dataDim = math.floor(len(data)/64.0)

		drawdown = numpy.zeros(int(dataDim)*poolSize, dtype=numpy.float32)
		drawdown_gpu = cuda.mem_alloc(drawdown.nbytes)
		cuda.memcpy_htod(drawdown_gpu, drawdown)

		calc = mod.get_function("calc")
		fitness = mod.get_function("fitness")

		calc(trees_gpu, data_gpu, winnerTable_gpu, numpy.int32(len(data)), block=(1, 64, 1), grid=(poolSize,int(dataDim))) # x is the tree index, y is the data index
		fitness(data_gpu, winnerTable_gpu, winCounter_gpu, lossCounter_gpu, timeInMarket_gpu, drawdown_gpu, numpy.int32(len(data)), block=(1, 64, 1), grid=(poolSize,int(dataDim))) 		 
		start = time.time()

		winCounter = numpy.empty_like(winCounter)
		cuda.memcpy_dtoh(winCounter, winCounter_gpu)

		lossCounter = numpy.empty_like(lossCounter)
		cuda.memcpy_dtoh(lossCounter, lossCounter_gpu)

		timeInMarket = numpy.empty_like(timeInMarket)
		cuda.memcpy_dtoh(timeInMarket, timeInMarket_gpu)

		drawdown = numpy.empty_like(drawdown)
		cuda.memcpy_dtoh(drawdown, drawdown_gpu)

		#print drawdown

		end = time.time()
		print "GPU Time", end - start
		
		evalArray = []
		#displayArr = []


		for x in range( poolSize ):
			#complexity = genome.getTreeComplexity(trees[x])

			winArray = winCounter[x*12: (x+1)*12]
			loseArray = lossCounter[x*12: (x+1)*12]
			#drawDownArray = drawdown[x*int(dataDim): (x+1)*int(dataDim)]
		
			#drawdownValue = drawDownArray.min()
			#drawdownValue = -1* drawdownValue
			#print drawdownValue

			stdDev = standardDeviation(winArray, loseArray)
			pf = profitFactor(winArray, loseArray)
			prof = profit(winArray, loseArray)
			marketTime = timeInMarket[x]

			evalArray.append( [pf, stdDev, prof, marketTime] )	
 		
		#print winCounter[0], lossCounter[0], timeInMarket[0]
		#print winCounter[1], lossCounter[1], timeInMarket[1]

		generations += 1

		print "Generation ", generations

		validWinnerTable = numpy.zeros(len(validData) * poolSize, dtype=numpy.bool)
		validWinnerTable_gpu = cuda.mem_alloc(validWinnerTable.nbytes)
		cuda.memcpy_htod(validWinnerTable_gpu, validWinnerTable)

		trees = numpy.array(trees).astype(numpy.float32)
		trees_gpu = cuda.mem_alloc(trees.nbytes)
		cuda.memcpy_htod(trees_gpu, trees)

		### gpu start
		start = time.time()

		validWinCounter = numpy.zeros(12 * poolSize, dtype=numpy.int32)
		validWinCounter_gpu = cuda.mem_alloc(validWinCounter.nbytes)
		cuda.memcpy_htod(validWinCounter_gpu, validWinCounter)

		validLossCounter = numpy.zeros(12 * poolSize, dtype=numpy.int32)
		validLossCounter_gpu = cuda.mem_alloc(validLossCounter.nbytes)
		cuda.memcpy_htod(validLossCounter_gpu, validLossCounter)

		validTimeInMarket = numpy.zeros(poolSize, dtype=numpy.int32)
		validTimeInMarket_gpu = cuda.mem_alloc(validTimeInMarket.nbytes)
		cuda.memcpy_htod(validTimeInMarket_gpu, validTimeInMarket)

		validDataDim = math.floor(len(validData)/64.0)
		

		validDrawdown = numpy.zeros(int(dataDim)*poolSize, dtype=numpy.float32)
		validDrawdown_gpu = cuda.mem_alloc(validDrawdown.nbytes)
		cuda.memcpy_htod(validDrawdown_gpu, validDrawdown)

		calc = mod.get_function("calc")
		fitness = mod.get_function("fitness")

		calc(trees_gpu, validData_gpu, validWinnerTable_gpu, numpy.int32(len(validData)), block=(1, 64, 1), grid=(poolSize, int(validDataDim))) # x is the tree index, y is the data index
		fitness(validData_gpu, validWinnerTable_gpu, validWinCounter_gpu, validLossCounter_gpu, validTimeInMarket_gpu, validDrawdown_gpu, numpy.int32(len(validData)), block=(1, 64, 1), grid=(poolSize, int(validDataDim))) 		 


		validWinCounter = numpy.empty_like(validWinCounter)
		cuda.memcpy_dtoh(validWinCounter, validWinCounter_gpu)

		validLossCounter = numpy.empty_like(validLossCounter)
		cuda.memcpy_dtoh(validLossCounter, validLossCounter_gpu)

		validTimeInMarket = numpy.empty_like(validTimeInMarket)
		cuda.memcpy_dtoh(validTimeInMarket, validTimeInMarket_gpu)


		for x in range( poolSize ):
			winArray = validWinCounter[x*12: (x+1)*12]
			loseArray = validLossCounter[x*12: (x+1)*12]

			#stdDev = standardDeviation(winArray, loseArray)
			
			pf = profitFactor(winArray, loseArray)
			marketTime = validTimeInMarket[x]

			
			evalArray[x].append( pf )	
			evalArray[x].append( marketTime )	

		pool.processGenomes(trees, evalArray)

		trees = pool.newPopulation(poolSize, treeLength)


	

