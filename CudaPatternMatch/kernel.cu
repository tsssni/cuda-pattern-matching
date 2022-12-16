#include "bf.cuh"
#include "kmp.cuh"
#include "bm.cuh"
#include "sunday.cuh"
#include "ssef.cuh"
#include "epsma.cuh"
#include "common.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <Windows.h>

char text[MAX_TEXT_LEN];
char pattern[MAX_PATTERN_LEN];
char* textDev;
char* patternDev;

int textLen;
int patternLen;
int threadTextLen;
int blockLen = 512;

int progCurr = 0;
int progTot;
LARGE_INTEGER freq;

void loadText(char* path)
{
	FILE* fp = fopen(path, "r");
	int ret = 0;
	int i = 0;
	
	while (ret != EOF)
	{
		ret = fscanf(fp, "%c", text + i);
		if (text[i] < 0)
		{
			continue;
		}

		++i;
	}

	textLen = strlen(text);
}

int patternMatchRand()
{
	srand(time(0));

	int r0 = rand() % 256;
	int r1 = (rand() % 256) << 8;
	int r2 = (rand() % 256) << 16;
	int r3 = ((rand() % 256) << 24) & (~(1 << 31));

	return r0 | r1 | r2 | r3;
}

void randomPattern()
{
	int start = patternMatchRand() % (textLen - patternLen);
	strncpy(pattern, text + start, patternLen);
	cudaMemcpy(patternDev, pattern, patternLen, cudaMemcpyHostToDevice);
}

void textInit(char* path)
{
	loadText(path);
	cudaMalloc(&textDev, textLen);
	cudaMemcpy(textDev, text, textLen, cudaMemcpyHostToDevice);
}

void patternInit(int initPatternLen)
{
	patternLen = initPatternLen;
	threadTextLen = 4 * patternLen;
	cudaMalloc(&patternDev, patternLen);
}

void progressBar()
{
	SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), { 0,0 });

	static char bar[200];
	int perc = (++progCurr) * 100 / progTot;
	int end;

	sprintf(bar, "%d%%", progCurr * 100 / progTot);
	end = strlen(bar);

	for (int i = 0; i < perc; ++i)
	{
		bar[end + i] = '#';
	}
	
	bar[end + perc] = '\0';
	printf("%s", bar);
}

double runtime(void (*func)())
{
	// 每次运行从文本中随机截取模式串
	// 求10000次运行时间的均值

	double avg = 0.0;

	for (int i = 0; i < 10000; ++i)
	{
		randomPattern();

		LARGE_INTEGER begin;
		LARGE_INTEGER end;

		QueryPerformanceCounter(&begin);
		func();
		QueryPerformanceCounter(&end);

		avg += (end.QuadPart - begin.QuadPart) * 1000.0 / (freq.QuadPart);
		progressBar();
	}

	avg /= 10000.0;
	return avg;
}

void run(char* path)
{
	void (*func[12])() =
	{
		bfCpu,bfGpu,
		kmpCpu,kmpGpu,
		bmCpu,bmGpu,
		sundayCpu,sundayGpu,
		epsmaCpu,epsmaGpu,
		ssefCpu,ssefGpu
	};

	char* funcName[] =
	{
		"Brute Force CPU","Brute Force GPU",
		"KMP CPU","KMP GPU",
		"BM CPU","BM GPU",
		"Sunday CPU","Sunday GPU",
		"EPSMa CPU","EPSMa GPU",
		"SSEF CPU","SSEF GPU"
	};

	int patternLenArr[] =
	{
		1 << 1,1 << 2,1 << 3,1 << 4,1 << 5,
		1 << 6,1 << 7,1 << 8,1 << 9,1 << 10
	};

	// 用于清空终端的空行
	char emptyLine[256];
	for (int i = 0; i < 255; ++i)
	{
		emptyLine[i] = ' ';
	}
	emptyLine[255] = '\0';

	textInit(path);
	char expPath[128] = "exp_";
	strcat(expPath, path);
	FILE* fp = fopen(expPath, "w");

	// 计算进度条总长度
	// 由于EPSMa CPU和SSEF CPU算法对模式串长度有限制
	// 故需要单独计算
	progTot = 10 * 10 * 10000 + 3 * 10000 + 6 * 10000;
	progCurr = 0;
	SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), { 0,0 });
	printf("%s", emptyLine);

	CONSOLE_CURSOR_INFO cursor_info = {1, 0};
	SetConsoleCursorInfo(GetStdHandle(STD_OUTPUT_HANDLE), &cursor_info);
	QueryPerformanceFrequency(&freq);

	for (int i = 0; i < 12; ++i)
	{
		for (int j = 0; j < 10; ++j)
		{
			// EPSMa CPU和SSEF CPU对模式串大小有限制
			if ((i == 8 && j >= 3) || (i == 10 && j <= 3))
			{
				continue;
			}
			
			patternInit(patternLenArr[j]);
			SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), { 0,1 });
			printf("Benchmark: %s\nAlgorithm: %s\nPattern Length: %d", path, funcName[i], patternLen);

			fprintf(fp, "func:%s,pattern length:%d,time:%.2f\n", funcName[i], patternLenArr[j], runtime(func[i]));
			cudaFree(patternDev);

			SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), { 0,1 });
			printf("%s\n%s\n%s", emptyLine, emptyLine, emptyLine);
		}
	}

	fclose(fp);
	cudaFree(textDev);
}

int main()
{
	// run("bible.txt");
	// run("gene.txt");
	return 0;
}