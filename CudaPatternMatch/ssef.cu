#include "ssef.cuh"
#include "common.cuh"
#include <stdlib.h>
#include <memory.h>
#include <math.h>
#include <immintrin.h>
#include <time.h>
#include <stdio.h>

#define ASCII_LEN 8
#define MAX_FILTER 65536

typedef struct SsefNode
{
	SsefNode* next = NULL;
	int idx = 1;
} SsefNode;

SsefNode filter[MAX_FILTER];
int filterLen;
int optimalOffset;
int validPatternLen;

void ssefGetOptimalOffsetCpu()
{
	// 统计最接近0、1各占50%的bit位
	// 使用此位生成哈希编码

	int bit[ASCII_LEN] = { 0 };

	for (int i = 0; i < patternLen; ++i)
	{
		for (int j = 0; j < ASCII_LEN; ++j)
		{
			bit[j] += (pattern[i] >> j) & 0x1;
		}
	}
	
	int minDist = INT_MAX;
	int optimalBit = 0;

	for (int i = 0; i < ASCII_LEN; ++i)
	{
		bit[i] = fabs(bit[i] - patternLen / 2);
		if (bit[i] < minDist)
		{
			minDist = bit[i];
			optimalBit = i;
		}
	}

	optimalOffset = ASCII_LEN - 1 - optimalBit;
}

void ssefBuildCpu()
{
	// 由于一个SIMD寄存器可容纳16个字符
	// 故哈希编码长度为16
	ssefGetOptimalOffsetCpu();
	filterLen = 16;
	validPatternLen = patternLen - 15;

	for (int i = 0; i < validPatternLen; ++i)
	{
		// 将每个字符的最佳比特位左移到符号位上
		// 由16个字符的符号位组成哈希编码
		auto simdPattern = _mm_loadu_si128((const __m128i*)&pattern[i]);
		auto tmp128 = _mm_slli_epi64(simdPattern, optimalOffset);
		auto f = _mm_movemask_epi8(tmp128);

		// 将该位置记录在链式哈希表对应的位置上
		SsefNode* node = &filter[f];
		while (node->next != NULL)
		{
			node = node->next;
		}

		node->next = (SsefNode*)malloc(sizeof(SsefNode));
		node->next->next = NULL;
		node->next->idx = i;
	}
}

void ssefFree()
{
	for (int i = 0; i < MAX_FILTER; ++i)
	{
		while (filter[i].next != NULL)
		{
			SsefNode* node = filter[i].next;
			filter[i].next = filter[i].next->next;
			free(node);
		}
	}
}

void ssefCpu()
{	
	// SSEF算法思路如下：
	// 由于SIMD寄存器最多容纳16个字符，故将文本按照16个字符分块，从0开始编号
	// 假设模式串长度为m，当m>=32,模式串一定可以占满某一块，最多占满n=floor(m/16)块
	// 可以证明模式串所占满的块中一定有编号为n-1的倍数的块
	// 我们每次都检查编号为n-1的倍数的块，利用最佳比特位构造哈希值查询已经构造好的哈希表
	// 若哈希表中存储了元素，则代表模式串中的某个位置的哈希值与该值相同
	// 接下来再进行逐字符比较即可

	ssefBuildCpu();
	int searchWindowLen = (floor(patternLen * 1.0 / filterLen) - 1) * filterLen;
	
	int* matchNum;
	int* matchIdx;
	patternMatchCpuMalloc(&matchNum, &matchIdx);
	
	for (int i = searchWindowLen; i < textLen; i += searchWindowLen)
	{
		__m128i simdText = _mm_loadu_si128((const __m128i*) & text[i]);
		__m128i tmp128 = _mm_slli_epi64(simdText, optimalOffset);
		int f = _mm_movemask_epi8(tmp128);

		SsefNode* node = filter[f].next;
		while (node != NULL)
		{
			if (node->idx == 0 || node->idx > searchWindowLen)
			{
				node = node->next;
				continue;
			}

			int j = i - node->idx;
			int k = 0;

			for (; k < patternLen; ++j, ++k)
			{
				if (text[j] != pattern[k])
				{
					break;
				}
			}

			if (k == patternLen)
			{
				matchIdx[*matchNum] = i - node->idx;
				++(*matchNum);
			}

			node = node->next;
		}
	}
	
#ifdef PRINT
	printMatchOutputCpu(matchNum, matchIdx);
#endif

	ssefFree();
	patternMatchCpuFree(matchNum, matchIdx);
}

void __global__ ssefGetOptimalOffsetKernel(const char* __restrict__ pattern, const int patternLen, int* bit)
{
	// 
	__shared__ int sharedBit[ASCII_LEN];
	extern __shared__ char sharedPattern[];
	int perThreadPatternLen = ceil(patternLen * 1.0 / blockDim.x);

	if (threadIdx.x < ASCII_LEN)
	{
		sharedBit[threadIdx.x] = 0;
	}

	for (int i = threadIdx.x * perThreadPatternLen; i < (threadIdx.x + 1) * perThreadPatternLen && i < patternLen; ++i)
	{
		sharedPattern[i] = pattern[i];
	}

	__syncthreads();

	int patternIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (patternIdx < patternLen)
	{
		for (int i = 0; i < ASCII_LEN; ++i)
		{
			atomicAdd(sharedBit + i, (sharedPattern[threadIdx.x] >> i) & 0x1);
		}

		if (threadIdx.x < ASCII_LEN)
		{
			atomicAdd(bit + threadIdx.x, sharedBit[threadIdx.x]);
		}
	}
}

void __device__ ssefAtomicLinkKernel(const int idx, SsefNode* filter, const int f)
{
	for (int i = 0; i < 32; ++i)
	{
		if (threadIdx.x % 32 != i)
		{
			continue;
		}

		while (atomicExch(&filter[f].idx, 0) == 0);

		SsefNode* node = &filter[f];
		while (node->next != NULL)
		{
			node = node->next;
		}

		node->next = (SsefNode*)malloc(sizeof(SsefNode));
		node->next->next = NULL;
		node->next->idx = idx;

		filter[f].idx = 1;
		return;
	}
}

void __global__ ssefBuildKernel(const char* __restrict__ pattern, const int patternLen, const int filterLen, const int optimalOffset, SsefNode* filter)
{
	extern __shared__ char sharedMovedPattern[];
	int offset = ASCII_LEN - 1 - optimalOffset;
	int validPatternLen = patternLen - filterLen + 1;

	int blockPatternIdx = blockIdx.x * blockDim.x;
	int blockPatternLen = blockDim.x + filterLen - 1;
	int perThreadPatternLen = ceil(blockPatternLen * 1.0 / blockDim.x);

	for (int i = threadIdx.x * perThreadPatternLen; i < (threadIdx.x + 1) * perThreadPatternLen && i < blockPatternLen && blockPatternIdx + i < patternLen; ++i)
	{
		sharedMovedPattern[i] = (pattern[blockPatternIdx + i] >> offset) & 0x1;
	}

	__syncthreads();

	int patternIdx = blockPatternIdx + threadIdx.x;
	if (patternIdx < validPatternLen)
	{
		int f = 0;
		for (int i = 0; i < filterLen; ++i)
		{
			f |= (sharedMovedPattern[threadIdx.x + i] << i);
		}

		ssefAtomicLinkKernel(patternIdx, filter, f);
	}
}

void __global__ ssefKernel(const char* __restrict__ text, const int textLen, const char* __restrict__ pattern, const int patternLen, const SsefNode* __restrict__ filter, const int filterLen, const int optimalOffset, int* matchNum, int* matchIdx)
{
	// 每个线程负责某个字符的移位运算并写入共享内存中
	extern __shared__ char sharedMemory[];
	char* sharedPattern = sharedMemory;
	char* sharedMovedText = sharedMemory + patternLen;
	int perThreadPatternLen = ceil(patternLen * 1.0 / blockDim.x);

	int offset = ASCII_LEN - 1 - optimalOffset;
	int validPatternLen = patternLen - filterLen + 1;
	int searchWindowLen = (floor(patternLen * 1.0 / filterLen) - 1) * filterLen;

	// 我们将线程按大小为16分组，每个线程移位运算结束后
	// 由该组的0号线程生成哈希值并查询哈希表
	// 为了提高效率采用交错分区的思想
	// 将编号相同的线程尽量集中在同一warp中执行
	// 否则每个warp32个线程最终只有2个线程在执行
	int windowsCnt = blockDim.x / filterLen;
	int windowIdx = threadIdx.x % windowsCnt;
	int windowThreadIdx = threadIdx.x / windowsCnt;

	int textIdx = (blockIdx.x*windowsCnt + windowIdx + 1) * searchWindowLen + windowThreadIdx;
	int blockTextIdx = windowIdx * filterLen + windowThreadIdx;
	
	for (int i = threadIdx.x * perThreadPatternLen; i < (threadIdx.x + 1) * perThreadPatternLen && i < patternLen; ++i)
	{
		sharedPattern[i] = pattern[i];
	}

	if (threadIdx.x < windowsCnt * filterLen)
	{
		sharedMovedText[blockTextIdx] = (text[textIdx] >> offset) & 0x1;
	}

	__syncthreads();
	
	if (windowThreadIdx == 0 && textIdx < textLen)
	{
		int f = 0;
		for (int i = 0; i < filterLen; ++i)
		{
			f |= sharedMovedText[blockTextIdx + i] << i;
		}
		
		SsefNode* node = filter[f].next;
		while (node != NULL)
		{
			if (node->idx == 0 || node->idx > searchWindowLen)
			{
				node = node->next;
				continue;
			}

			int j = textIdx - node->idx;
			int k = 0;

			for (; k < patternLen; ++j, ++k)
			{
				if (text[j] != pattern[k])
				{
					break;
				}
			}

			if (k == patternLen)
			{
				int idx = atomicAdd(matchNum, 1);
				matchIdx[idx] = textIdx - node->idx;
			}

			node = node->next;
		}
	}
}

void __global__ ssefFreeKernel(SsefNode* filter)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	while (idx < MAX_FILTER && filter[idx].next != NULL)
	{
		SsefNode* node = filter[idx].next;
		filter[idx].next = filter[idx].next->next;
		free(node);
	}
}

void ssefGetOptimalOffsetGpu()
{
	dim3 blockSize = blockLen;
	dim3 gridSize = ceil(patternLen * 1.0 / blockSize.x);

	int* bit = (int*)malloc(ASCII_LEN * sizeof(int));
	int* bitDev = nullptr;

	cudaMalloc(&bitDev, ASCII_LEN * sizeof(int));
	cudaMemset(bitDev, 0, ASCII_LEN * sizeof(int));
	ssefGetOptimalOffsetKernel <<< gridSize, blockSize, patternLen >>> (patternDev, patternLen, bitDev);
	cudaMemcpy(bit, bitDev, ASCII_LEN * sizeof(int), cudaMemcpyDeviceToHost);

	int minDist = INT_MAX;
	int optimalBit = 0;

	for (int i = 0; i < ASCII_LEN; ++i)
	{
		bit[i] = fabs(bit[i] - patternLen / 2);
		if (bit[i] < minDist)
		{
			minDist = bit[i];
			optimalBit = i;
		}
	}

	optimalOffset = ASCII_LEN - 1 - optimalBit;
}

void ssefBuildGpu(SsefNode** filterDevPtr)
{
	ssefGetOptimalOffsetGpu();

	dim3 blockSize = blockLen;
	dim3 gridSize = ceil(patternLen * 1.0 / blockSize.x);

	filterLen = patternLen > 32 ? 16 : patternLen / 2;
	cudaMalloc(filterDevPtr, MAX_FILTER * sizeof(SsefNode));
	SsefNode* filterDev = *filterDevPtr;

	cudaMemcpy(filterDev, filter, MAX_FILTER * sizeof(SsefNode), cudaMemcpyHostToDevice);
	ssefBuildKernel <<< gridSize, blockSize, blockSize.x + filterLen - 1 >>> (patternDev, patternLen, filterLen, optimalOffset, filterDev);
}

void ssefGpu()
{
	SsefNode* filterDev;
	ssefBuildGpu(&filterDev);

	int searchWindowLen = (floor(patternLen * 1.0 / filterLen) - 1) * filterLen;
	int windowsCnt = textLen / searchWindowLen;
	int blockWindowsCnt = blockLen / filterLen;

	dim3 blockSize = blockLen;
	dim3 gridSize = ceil(windowsCnt * 1.0 / blockWindowsCnt);

	int* matchNumDev;
	int* matchIdxDev;
	patternMatchGpuMalloc(&matchNumDev, &matchIdxDev);

	ssefKernel <<< gridSize, blockSize, patternLen + blockWindowsCnt * filterLen >>> (textDev, textLen, patternDev, patternLen, filterDev, filterLen, optimalOffset, matchNumDev, matchIdxDev);

#ifdef PRINT
	printMatchOutputGpu(matchNumDev, matchIdxDev);
#endif
	
	gridSize = ceil(MAX_FILTER * 1.0 / blockLen);
	ssefFreeKernel <<< gridSize, blockSize >>> (filterDev);
	cudaFree(filterDev);
	patternMatchGpuFree(matchNumDev, matchIdxDev);
}