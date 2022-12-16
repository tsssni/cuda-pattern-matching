#include "bm.cuh"
#include "common.cuh"
#include <math.h>
#include <stdlib.h>
#include <memory.h>

void bmBuildCpu(int** badCharPtr, int** goodSuffixPtr)
{
	*badCharPtr = (int*)malloc(ALPHABET_SIZE * sizeof(int));
	*goodSuffixPtr = (int*)malloc(patternLen * sizeof(int));
	int* maxSuffix = (int*)malloc(patternLen * sizeof(int));

	int* badChar = *badCharPtr;
	int* goodSuffix = *goodSuffixPtr;
	memset(badChar, 0xff, sizeof(int) * ALPHABET_SIZE);
	memset(goodSuffix, 0xff, sizeof(int) * patternLen);

	// 从前到后顺序遍历，将对应字符位置的badChar值修改为当前i值
	// 通过这种方法保证badChar中存储的是某个字符最后出现的位置
	// 匹配过程中跳转时可以避免应跳转长度过长导致丢失匹配
	for (int i = 0; i < patternLen; ++i)
	{
		badChar[pattern[i]] = i;
	}

	// 在maxSuffix中的每个位置i记录的值代表与子串[0-i]的后缀匹配的
	// 最长模式串后缀的长度
	maxSuffix[patternLen - 1] = patternLen;
	for (int i = 0; i < patternLen - 1; ++i)
	{
		int j = i;
		int k = patternLen - 1;
		int cnt = 0;

		while (j >= 0 && pattern[j]==pattern[k])
		{
			++cnt;
			--j;
			--k;
		}

		maxSuffix[i] = cnt;
	}

	// 通过maxSuffix中记录的值可以很容易的计算出
	// 模式串中每个后缀上一次在模式串中出现的位置
	// 若没有出现则为初始化值-1
	// 记录最后一次出现位置的原因与坏字符数组构建过程相同
	for (int i = 0; i < patternLen - 1; ++i)
	{
		if (maxSuffix[i] > 0)
		{
			goodSuffix[patternLen - maxSuffix[i]] = i;
		}
	}

	free(maxSuffix);
}

void bmCpu()
{
	int* badChar;
	int* goodSuffix;
	bmBuildCpu(&badChar, &goodSuffix);

	int* matchNum;
	int* matchIdx;
	patternMatchCpuMalloc(&matchNum, &matchIdx);

	int i = patternLen - 1;

	while (i < textLen)
	{
		int j = patternLen - 1;
		int k = i;
		int len = 0;

		// 匹配时从后向前匹配
		while (j>=0 && text[k] == pattern[j])
		{
			--k;
			--j;
		}

		if (j < 0)
		{
			++k;
			matchIdx[*matchNum] = k;
			++(*matchNum);
			++i;
		}
		else
		{
			// 若匹配失败，则使用坏字符跳转策略与好后缀跳转策略
			// 选取跳转长度较大者执行跳转
			// 若二者跳转长度均小于1则跳转长度设为1

			// 好字符策略将当前位置与当前文本位置的字符在模式串中最后一次出现的位置
			// 相减作为跳转长度，这样跳转之后的模式串字符与当前文本位置的字符相等
			// 再次由后向前匹配即可
			// 这种情况向跳转位置可能为负数，这里限制跳转最小长度为1
			int largestGoodSuffixPos = -1;
			int offset = __max(1, j - badChar[text[k]]);

			// 好后缀策略处理已经匹配成功的字符，由于是由后向前匹配
			// 故匹配成功的字符串是模式串的后缀
			// 若后缀在之前的模式串中未出现，则缩短后缀长度直到出现为止
			// 将模式串尾部位置减去后缀上一次出现位置作为跳转长度
			// 这样可以保证之前已经匹配的后缀没有被浪费
			// 若没有一个后缀出现过则相当于没有对跳转位置进行处理
			for (int l = j + 1; l < patternLen - 1; ++l)
			{
				if (goodSuffix[l] != -1)
				{
					offset = __max(offset, patternLen - 1 - goodSuffix[l]);
					break;
				}
			}

			i += offset;
		}
	}

#ifdef PRINT
	printMatchOutputCpu(matchNum, matchIdx);
#endif

	free(badChar);
	free(goodSuffix);
	patternMatchCpuFree(matchNum, matchIdx);
}

void __global__ bmBadCharKernel(const char* __restrict__ pattern, const int patternLen, int* badChar)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < patternLen)
	{
		// 并行计算字符出现的最大位置时防止写入冲突
		// 这里不需要使用私有原子操作，因为只进行一次写入
		atomicMax(badChar + pattern[idx], idx);
	}
}

void __global__ bmGoodSuffixKernel(const char* __restrict__ pattern, const int patternLen, int* goodSuffix)
{
	extern __shared__ char sharedPattern[];
	int perThreadPatternLen = ceil(patternLen * 1.0 / blockDim.x);

	for (int i = threadIdx.x * perThreadPatternLen; i < (threadIdx.x + 1) * perThreadPatternLen && i < patternLen; ++i)
	{
		sharedPattern[i] = pattern[i];
	}

	__syncthreads();

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < patternLen - 1)
	{
		int cnt = 0;
		int j = idx;
		int k = patternLen - 1;

		while (j >= 0 && sharedPattern[j] == sharedPattern[k])
		{
			++cnt;
			--j;
			--k;
		}

		if (cnt > 0)
		{
			// 统计出子串[0,idx]与模式串后缀最大匹配长度之后
			// 立即进行比较并执行原子写入，无需等待maxSuffix全部计算完毕
			// 模式串长度较小，使用私有原子操作提升不大
			atomicMax(goodSuffix + patternLen - cnt, idx);
		}
	}
}

void __global__ bmKernel(const char* __restrict__ text, const int textLen, const int threadTextLen, const char* __restrict__ pattern, const int patternLen, const int* __restrict__ badChar, const int* __restrict__ goodSuffix, int* matchNum, int* matchIdx)
{
	// 由于每个线程都负责处理一段文本，共享内存无法容纳每个block所需处理文本，故不将文本加载进共享内存
	// 由于block处理文本量大，理论匹配量也较大，故无法容纳私有化原子操作
	__shared__ int sharedBadChar[ALPHABET_SIZE];
	extern __shared__ char sharedMemory[];
	int* sharedGoodSuffix = (int*)(sharedMemory);
	char* sharedPattern = (char*)(sharedGoodSuffix + patternLen);

	int perThreadBadCharLen = ceil(ALPHABET_SIZE * 1.0 / blockDim.x);
	int perThreadPatternLen = ceil(patternLen * 1.0 / blockDim.x);

	for (int i = threadIdx.x * perThreadBadCharLen; i < (threadIdx.x + 1) * perThreadBadCharLen && i < ALPHABET_SIZE; ++i)
	{
		sharedBadChar[i] = badChar[i];
	}

	for (int i = threadIdx.x * perThreadPatternLen; i < (threadIdx.x + 1) * perThreadPatternLen && i < patternLen; ++i)
	{
		sharedPattern[i] = pattern[i];
		sharedGoodSuffix[i] = goodSuffix[i];
	}

	__syncthreads();

	// 为确保找到所有匹配，每个线程额外检查模式串长度-1个字符
	int extendedThreadTextLen = threadTextLen + patternLen - 1;
	int textIdx = (blockIdx.x * blockDim.x + threadIdx.x) * threadTextLen;
	int i = patternLen - 1;

	while (i < extendedThreadTextLen && textIdx + i < textLen)
	{
		int j = patternLen - 1;
		int k = i;
		int len = 0;
		
		while (j >= 0 && text[textIdx + k] == sharedPattern[j])
		{
			--k;
			--j;
		}

		if (j < 0)
		{
			++k;
			int idx = atomicAdd(matchNum, 1);
			matchIdx[idx] = textIdx + k;
			++i;
		}
		else
		{
			int largestGoodSuffixPos = -1;
			int offset = __max(1, j - sharedBadChar[text[textIdx + k]]);

			for (int l = j + 1; l < patternLen - 1; ++l)
			{
				if (sharedGoodSuffix[l] != -1)
				{
					offset = __max(offset, patternLen - 1 - sharedGoodSuffix[l]);
					break;
				}
			}

			i += offset;
		}
	}
}

void bmBuildGpu(int** badCharDevPtr, int** goodSuffixDevPtr)
{
	dim3 blockSize = blockLen;
	dim3 gridSize = ceil(patternLen * 1.0 / blockSize.x);

	cudaMalloc(badCharDevPtr, sizeof(int) * ALPHABET_SIZE);
	cudaMemset(*badCharDevPtr, 0xff, sizeof(int) * ALPHABET_SIZE);
	bmBadCharKernel <<< gridSize, blockSize >>> (patternDev, patternLen, *badCharDevPtr);
	
	cudaMalloc(goodSuffixDevPtr, sizeof(int) * patternLen);
	cudaMemset(*goodSuffixDevPtr, 0xff, sizeof(int) * patternLen);
	bmGoodSuffixKernel <<< gridSize, blockSize, patternLen >>> (patternDev, patternLen, *goodSuffixDevPtr);	
}

void bmGpu()
{
	dim3 blockSize = blockLen;
	dim3 gridSize = ceil(textLen * 1.0 / (blockSize.x * threadTextLen));

	int* matchNumDev = nullptr;
	int* matchIdxDev = nullptr;
	patternMatchGpuMalloc(&matchNumDev, &matchIdxDev);

	int* badCharDev = nullptr;
	int* goodSuffixDev = nullptr;
	bmBuildGpu(&badCharDev, &goodSuffixDev);

	bmKernel <<< gridSize, blockSize, patternLen * (sizeof(char) + sizeof(int)) >>> (textDev, textLen, threadTextLen, patternDev, patternLen, badCharDev, goodSuffixDev, matchNumDev, matchIdxDev);

#ifdef PRINT
	printMatchOutputGpu(matchNumDev, matchIdxDev);
#endif

	cudaFree(badCharDev);
	cudaFree(goodSuffixDev);
	patternMatchGpuFree(matchNumDev, matchIdxDev);
}
