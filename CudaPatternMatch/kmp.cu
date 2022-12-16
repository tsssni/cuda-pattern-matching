#include "kmp.cuh"
#include "common.cuh"
#include <stdlib.h>
#include <math.h>

void kmpNextCpu(int** nextPtr)
{
	// next数组记录与当前位置之前的子串的后缀相同的模式串前缀的下一个位置
	*nextPtr = (int*)malloc(patternLen * sizeof(int));
	int* next = *nextPtr;
	int i = 0;
	int j = -1;
	next[0] = -1;

	while (i < patternLen - 1)
	{
		if (j == -1 || pattern[i] == pattern[j])
		{
			++i, ++j;
			if (pattern[i] != pattern[j])
			{
				// 由于模式串中子串[i-j,i-1]与前缀[0,j-1]相同
				// 故j即为与模式串在i位置之前的子串的后缀相等的模式串前缀的后一个字符
				next[i] = j;
			}
			else
			{
				// 若字符相等
				// 则此时子串[i-j,i-1]与子串[0,j-1]相等
				// 将j赋给next[i]是符合要求的
				// 但跳转之后由于i与j位置字符相等则与文本字符仍然不匹配
				// 故将next[j]赋给next[i]来进行优化
				next[i] = next[j];
			}
		}
		else
		{
			// 模式串不等，将j滑动至next[j]继续尝试
			j = next[j];
		}
	}
}

void kmpCpu()
{
	int* next = nullptr;
	int i = 0;
	int j = 0;
	kmpNextCpu(&next);

	int* matchNum;
	int* matchIdx;
	patternMatchCpuMalloc(&matchNum, &matchIdx);

	while (i < textLen)
	{
		if (j == -1 || text[i] == pattern[j])
		{
			++i, ++j;
		}
		else
		{
			j = next[j];
		}

		if (j == patternLen)
		{
			i -= patternLen;
			j = -1;
			matchIdx[*matchNum] = i;
			++(*matchNum);
		}
	}

#ifdef PRINT
	printMatchOutputCpu(matchNum, matchIdx);
#endif
	
	free(next);
	patternMatchCpuFree(matchNum, matchIdx);
}

void __global__ kmpKernel(const char* __restrict__ text, const int textLen, const int threadTextLen, const char* __restrict__ pattern, const int patternLen, const int* __restrict__ next, int* matchNum, int* matchIdx)
{
	// 由于每个线程都负责处理一段文本，共享内存无法容纳每个block所需处理文本，故不将文本加载进共享内存
	// 由于block处理文本量大，理论匹配量也较大，故无法容纳私有化原子操作
	extern __shared__ char sharedMemory[];
	int* sharedNext = (int*)sharedMemory;
	char* sharedPattern = (char*)(sharedNext + patternLen);
	int perThreadPatternLen = ceil(patternLen * 1.0 / blockDim.x);

	for (int i = threadIdx.x * perThreadPatternLen; i < (threadIdx.x + 1) * perThreadPatternLen && i < patternLen; ++i)
	{
		sharedPattern[i] = pattern[i];
		sharedNext[i] = next[i];
	}

	__syncthreads();

	// 为确保找到所有匹配，每个线程额外检查模式串长度-1个字符
	int extendedThreadTextLen = threadTextLen + patternLen - 1;
	int textIdx = (blockIdx.x * blockDim.x + threadIdx.x) * threadTextLen;

	int i = 0;
	int j = 0;
	
	while (i < extendedThreadTextLen && textIdx + i < textLen)
	{
		if (j == -1 || text[textIdx + i] == sharedPattern[j])
		{
			++i, ++j;
		}
		else
		{
			j = sharedNext[j];
		}

		if (j == patternLen)
		{
			i -= patternLen;
			j = -1;

			int idx = atomicAdd(matchNum, 1);
			matchIdx[idx] = textIdx + i;
		}
	}
}

void kmpGpu()
{
	dim3 blockSize = blockLen;
	dim3 gridSize = ceil(textLen * 1.0 / (blockSize.x * threadTextLen));

	int* next = nullptr;
	int* nextDev = nullptr;

	// 由于next数组后生成的元素依赖于已生成的元素，故不使用并行化加速
	kmpNextCpu(&next);
	cudaMalloc(&nextDev, sizeof(int) * patternLen);
	cudaMemcpy(nextDev, next, sizeof(int) * patternLen, cudaMemcpyHostToDevice);
	free(next);

	int* matchNumDev = nullptr;
	int* matchIdxDev = nullptr;
	patternMatchGpuMalloc(&matchNumDev, &matchIdxDev);

	kmpKernel <<< gridSize, blockSize, patternLen * (sizeof(char) + sizeof(int)) >>> (textDev, textLen, threadTextLen, patternDev, patternLen, nextDev, matchNumDev, matchIdxDev);

#ifdef PRINT
	printMatchOutputGpu(matchNumDev, matchIdxDev);
#endif

	cudaFree(nextDev);
	patternMatchGpuFree(matchNumDev, matchIdxDev);
}
