#include "common.cuh"
#include <cstdio>
#include <algorithm>

void patternMatchCpuMalloc(int** matchNumPtr, int** matchIdxPtr)
{
	*matchNumPtr = (int*)malloc(sizeof(int));
	**matchNumPtr = 0;
	*matchIdxPtr = (int*)malloc(textLen * sizeof(int));
}

void patternMatchGpuMalloc(int** matchNumDevPtr, int** matchIdxDevPtr)
{
	cudaMalloc(matchNumDevPtr, sizeof(int));
	cudaMalloc(matchIdxDevPtr, sizeof(int) * textLen);
	cudaMemset(*matchNumDevPtr, 0, sizeof(int));
}

void patternMatchCpuFree(int* matchNum, int* matchIdx)
{
	free(matchNum);
	free(matchIdx);
}

void patternMatchGpuFree(int* matchNumDev, int* matchIdxDev)
{
	cudaFree(matchNumDev);
	cudaFree(matchIdxDev);
}

void printMatchOutputCpu(int* matchNum, int* matchIdx)
{
	for (int i = 0; i < *matchNum; ++i)
	{
		printf("%d ", matchIdx[i]);
	}

	putchar('\n');
}

void printMatchOutputGpu(int* matchNumDev, int* matchIdxDev)
{
	int* matchNum = (int*)malloc(sizeof(int));
	cudaMemcpy(matchNum, matchNumDev, sizeof(int), cudaMemcpyDeviceToHost);
	int* matchIdx = (int*)malloc((*matchNum) * sizeof(int));
	cudaMemcpy(matchIdx, matchIdxDev, sizeof(int) * (*matchNum), cudaMemcpyDeviceToHost);
	std::sort(matchIdx, matchIdx + *matchNum);

	for (int i = 0; i < *matchNum; ++i)
	{
		printf("%d ", matchIdx[i]);
	}

	putchar('\n');

	free(matchNum);
	free(matchIdx);
}