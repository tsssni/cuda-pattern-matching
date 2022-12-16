#include "sunday.cuh"
#include "common.cuh"
#include <stdlib.h>
#include <memory.h>
#include <math.h>

void sundayMoveCpu(int** movePtr)
{
	*movePtr = (int*)malloc(ALPHABET_SIZE * sizeof(int));
	int* move = *movePtr;
	memset(move, 0xff, sizeof(int) * ALPHABET_SIZE);

	// 从前到后顺序遍历，将对应字符位置的move值修改为当前i值
	// 通过这种方法保证move中存储的是某个字符最后出现的位置
	// 匹配过程中跳转时可以避免应跳转长度过长导致丢失匹配
	for (int i = 0; i < patternLen; ++i)
	{
		move[pattern[i]] = i;
	}
}

void sundayCpu()
{
	int* move;
	sundayMoveCpu(&move);

	int* matchNum;
	int* matchIdx;
	patternMatchCpuMalloc(&matchNum,&matchIdx);

	int i = 0;

	while (i <= textLen - patternLen)
	{
		int j = 0;
		int k = i;

		while (j < patternLen && text[k] == pattern[j])
		{
			++k;
			++j;
		}

		if (j == patternLen)
		{
			matchIdx[*matchNum] = i;
			++(*matchNum);

			i += 1;
		}
		else
		{
			// 若匹配失败，则说明子串[i,i+patternLen-1]与模式串匹配失败
			// 此时检查当前文本位置加上模式串长度之后的字符，即i+patternLen位置的字符
			// 因为最近的可能成功的匹配是[i+1,i+patternLen]
			// 检查i+patternLen位置的字符在模式串中最后一次出现的位置并执行跳转
			// 若不存在则该跳转方法可以直接将i跳转到i+patternLen+1，效率很高

			int lastPos = move[text[i + patternLen]];
			int offset = patternLen - lastPos;
			i += offset;
		}
	}

#ifdef PRINT
	printMatchOutputCpu(matchNum, matchIdx);
#endif

	free(move);
	patternMatchCpuFree(matchNum, matchIdx);
}

void __global__ sundayMoveKernel(const char* __restrict__ pattern, const int patternLen, int* move)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < patternLen)
	{
		// 并行计算字符出现的最大位置时防止写入冲突
		// 这里不需要使用私有原子操作，因为只进行一次写入
		atomicMax(move + pattern[idx], idx);
	}
}

void __global__ sundayKernel(const char* __restrict__ text, const int textLen, const int threadTextLen, const char* __restrict__ pattern, const int patternLen, const int* __restrict__ move, int* matchNum, int* matchIdx)
{
	// 由于每个线程都负责处理一段文本，共享内存无法容纳每个block所需处理文本，故不将文本加载进共享内存
	// 由于block处理文本量大，理论匹配量也较大，故无法容纳私有化原子操作
	__shared__ int sharedMove[ALPHABET_SIZE];
	extern __shared__ char sharedPattern[];
	int perThreadMoveLen = ceil(ALPHABET_SIZE * 1.0 / blockDim.x);
	int perThreadPatternLen = ceil(patternLen * 1.0 / blockDim.x);
	
	for (int i = threadIdx.x * perThreadMoveLen; i < (threadIdx.x + 1) * perThreadMoveLen && i < ALPHABET_SIZE; ++i)
	{
		sharedMove[i] = move[i];
	}

	for (int i = threadIdx.x * perThreadPatternLen; i < (threadIdx.x + 1) * perThreadPatternLen && i < patternLen; ++i)
	{
		sharedPattern[i] = pattern[i];
	}

	__syncthreads();

	// 为确保找到所有匹配，每个线程额外检查模式串长度-1个字符
	int extendedThreadTextLen = threadTextLen + patternLen - 1;
	int textIdx = (blockIdx.x * blockDim.x + threadIdx.x) * threadTextLen;
	int i = 0;

	while (i < threadTextLen && textIdx + i < textLen)
	{
		int j = 0;
		int k = i;

		while (j < patternLen && text[textIdx + k] == sharedPattern[j])
		{
			++k;
			++j;
		}

		if (j == patternLen)
		{
			int idx = atomicAdd(matchNum, 1);
			matchIdx[idx] = textIdx + i;

			i += 1;
		}
		else
		{
			int lastPos = sharedMove[text[textIdx + i + patternLen]];
			int offset = patternLen - lastPos;
			i += offset;
		}
	}
}



void sundayMoveGpu(int** movePtr)
{
	dim3 blockSize = blockLen;
	dim3 gridSize = ceil(patternLen * 1.0 / blockSize.x);

	cudaMalloc(movePtr, sizeof(int) * ALPHABET_SIZE);
	cudaMemset(*movePtr, 0xff, sizeof(int) * ALPHABET_SIZE);
	sundayMoveKernel <<< gridSize, blockSize, patternLen >>> (patternDev, patternLen, *movePtr);
}

void sundayGpu()
{
	dim3 blockSize = blockLen;
	dim3 gridSize = ceil(textLen * 1.0 / (blockSize.x * threadTextLen));

	int* matchNumDev = nullptr;
	int* matchIdxDev = nullptr;
	patternMatchGpuMalloc(&matchNumDev, &matchIdxDev);

	int* moveDev = nullptr;
	sundayMoveGpu(&moveDev);

	sundayKernel <<< gridSize, blockSize, patternLen >>> (textDev, textLen, threadTextLen, patternDev, patternLen, moveDev, matchNumDev, matchIdxDev);

#ifdef PRINT
	printMatchOutputGpu(matchNumDev, matchIdxDev);
#endif

	cudaFree(moveDev);
	patternMatchGpuFree(matchNumDev, matchIdxDev);
}
