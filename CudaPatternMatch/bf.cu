#include "bf.cuh"
#include "common.cuh"
#include <math.h>

void bfCpu()
{
	int* matchNum;
	int* matchIdx;
	patternMatchCpuMalloc(&matchNum, &matchIdx);

	for (int i = 0; i < textLen - patternLen; ++i)
	{
		int j = 0;
		int k = i;

		for (; j < patternLen; ++j, ++k)
		{
			if (text[k] != pattern[j])
			{
				break;
			}
		}

		if (j == patternLen)
		{
			matchIdx[*matchNum] = i;
			++(*matchNum);
		}
	}

#ifdef PRINT
	printMatchOutputCpu(matchNum, matchIdx);
#endif

	patternMatchCpuFree(matchNum, matchIdx);
}

void __global__ bfKernel(const char* __restrict__ text, const int textLen, const char* __restrict__ pattern, const int patternLen, int* matchNum, int* matchIdx)
{
	// 私有化原子操作加速
	// 由于共享内存足够容纳每个block负责处理的文本串，故加载进共享内存
	__shared__ int sharedMatchNum;
	__shared__ int sharedWriteIdx;
	extern __shared__ char sharedMemory[];
	int* sharedMatchIdx = (int*)sharedMemory;
	char* sharedPattern = (char*)(sharedMatchIdx + blockDim.x);
	char* sharedText = sharedPattern + patternLen;
	
	// 为确保检查到所有匹配，每个block需要额外检查模式串长度-1个字符
	sharedMatchNum = 0;
	int blockTextIdx = blockIdx.x * blockDim.x;
	int blockTextLen = blockDim.x + patternLen - 1; 	
	int perThreadPatternLen = ceil(patternLen * 1.0 / blockDim.x);
	int perThreadTextLen = ceil(blockTextLen * 1.0 / blockDim.x);

	for (int i = threadIdx.x * perThreadPatternLen; i < (threadIdx.x + 1) * perThreadPatternLen && i < patternLen; ++i)
	{
		sharedPattern[i] = pattern[i];
	}
	
	for (int i = threadIdx.x * perThreadTextLen; i < (threadIdx.x + 1) * perThreadTextLen && i < blockTextLen && blockTextIdx + i < textLen; ++i)
	{
		sharedText[i] = text[blockTextIdx + i];
	}

	__syncthreads();
	
	int textIdx = blockTextIdx + threadIdx.x;
	if (textIdx <= textLen - patternLen)
	{
		int i = 0;
		for (; i < patternLen; ++i)
		{
			if (sharedPattern[i] != sharedText[threadIdx.x + i])
			{
				break;
			}
		}

		if (i == patternLen)
		{
			int idx = atomicAdd(&sharedMatchNum, 1);
			sharedMatchIdx[idx] = textIdx;
		}
	}

	__syncthreads();

	if (threadIdx.x < sharedMatchNum)
	{
		// 线程号小于block内部匹配数量的线程负责最终拷贝
		// 0号线程负责最终原子操作
		if (threadIdx.x == 0)
		{
			sharedWriteIdx = atomicAdd(matchNum, sharedMatchNum);
		}

		__syncthreads();

		matchIdx[sharedWriteIdx + threadIdx.x] = sharedMatchIdx[threadIdx.x];
	}
}

void bfGpu()
{
	dim3 blockSize = blockLen;
	dim3 gridSize = ceil(textLen * 1.0 / blockSize.x);
	int blockTextLen = blockSize.x + patternLen - 1;

	int* matchNumDev = nullptr;
	int* matchIdxDev = nullptr;
	patternMatchGpuMalloc(&matchNumDev, &matchIdxDev);

	bfKernel <<< gridSize, blockSize, patternLen + blockTextLen + blockSize.x * sizeof(int) >>>  (textDev, textLen, patternDev, patternLen, matchNumDev, matchIdxDev);

#ifdef PRINT
	printMatchOutputGpu(matchNumDev, matchIdxDev);
#endif

	patternMatchGpuFree(matchNumDev, matchIdxDev);
}
