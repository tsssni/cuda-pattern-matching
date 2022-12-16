#include "epsma.cuh"
#include "common.cuh"
#include <stdlib.h>
#include <memory.h>
#include <math.h>
#include <immintrin.h>

void epsmaBuildCpu(char** extendedPtr)
{
	// 将模式串每个字符复制16次用于SIMD比较
	// 例如将模式串"ab"扩展为
	// "aaaaaaaaaaaaaaaabbbbbbbbbbbbbbbb"

	*extendedPtr = (char*)malloc(patternLen * 16);
	char* extended = *extendedPtr;

	for (int i = 0; i < patternLen; ++i)
	{
		memset(extended + i * 16, pattern[i], 16);
	}
}

void epsmaCpu()
{
	char* extended;
	epsmaBuildCpu(&extended);
	
	int* matchNum;
	int* matchIdx;
	patternMatchCpuMalloc(&matchNum, &matchIdx);

	for (int i = 0; i < textLen; i += 16 - patternLen + 1)
	{
		auto simdText = _mm_loadu_si128((const __m128i*)(text + i));
		int matchRes = 0xffff;

		for (int j = 0; j < patternLen; ++j)
		{
			// 将模式串每个字符复制16次并存入SIMD寄存器
			// 与当前扫描到的16个存入SIMD寄存器的文本字符进行比较
			// 比较结果用16bit存储，并右移j位与上一次结果做与运算
			// 循环结束之后若比较结果不为0则说明匹配成功

			// 例：ab与abcdacccddddabac比较
			// 第一次比较将a*16与文本比较
			// 结果为0101000000010001（该结果为左边为高位比较结果，右边为低位比较结果）
			// 第二次比较将b*16与文本比较
			// 结果为0010000000000010，右移一位与上一次结果相与得：0001000000000001
			
			// 可以看出匹配的原理是第n为的比较结果右移n位，若经过patternLen次比较后
			// 该位置仍然是1，则说明该位置往后patternLen个字符与模式串相匹配
			// 由于每轮比较都需要额外的字符，例如比较6个位置的匹配，模式串长度为2
			// 则共需要7个字符，否则第6个字符无法进行匹配
			// 故为了充分利用SIMD寄存器的空间，模式串长度最大为16

			auto simdPattern = _mm_loadu_si128((const __m128i*)(extended + j * 16));
			auto cmpRes = _mm_cmpeq_epi8(simdText, simdPattern);
			matchRes &= _mm_movemask_epi8(cmpRes) >> j;

			if (!matchRes)
			{
				break;
			}
		}

		if (matchRes)
		{
			for (int j = 0; j < 16 - patternLen + 1; ++j)
			{
				if (((matchRes >> j) & 0x1) && i + j < textLen - patternLen)
				{
					matchIdx[*matchNum] = i + j;
					++(*matchNum);
				}
			}
		}
	}

#ifdef PRINT
	printMatchOutputCpu(matchNum, matchIdx);
#endif

	free(extended);
	patternMatchCpuFree(matchNum, matchIdx);
}

void __global__ epsmaKernel(const char* __restrict__ text, const int textLen, const char* __restrict__ pattern, const int patternLen, int* matchNum, int* matchIdx)
{
	__shared__ int sharedMatchNum;
	__shared__ int sharedWriteIdx;
	extern __shared__ char sharedMemory[];
	int* sharedMatchIdx = (int*)sharedMemory;
	char* sharedPattern = (char*)(sharedMatchIdx + blockDim.x);
	char* sharedText = sharedPattern + patternLen;

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
		if (threadIdx.x == 0)
		{
			sharedWriteIdx = atomicAdd(matchNum, sharedMatchNum);
		}

		__syncthreads();

		matchIdx[sharedWriteIdx + threadIdx.x] = sharedMatchIdx[threadIdx.x];
	}

}

void epsmaGpu()
{
	dim3 blockSize = blockLen;
	dim3 gridSize = ceil(textLen * 1.0 / blockSize.x);
	int blockTextLen = blockSize.x + patternLen - 1;

	int* matchNumDev = nullptr;
	int* matchIdxDev = nullptr;
	patternMatchGpuMalloc(&matchNumDev, &matchIdxDev);

	epsmaKernel <<< gridSize, blockSize, patternLen + blockTextLen + blockSize.x * sizeof(int) >>>  (textDev, textLen, patternDev, patternLen, matchNumDev, matchIdxDev);

#ifdef PRINT
	printMatchOutputGpu(matchNumDev, matchIdxDev);
#endif

	patternMatchGpuFree(matchNumDev, matchIdxDev);
}