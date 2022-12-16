#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#define ALPHABET_SIZE 256
#define MAX_TEXT_LEN 5242800
#define MAX_PATTERN_LEN 2048

extern char text[MAX_TEXT_LEN];
extern char pattern[MAX_PATTERN_LEN];
extern char* textDev;
extern char* patternDev;
extern int textLen;
extern int patternLen;
extern int threadTextLen;
extern int blockLen;

void patternMatchCpuMalloc(int** matchNumPtr, int** matchIdxPtr);
void patternMatchGpuMalloc(int** matchNumDevPtr, int** matchIdxDevPtr);
void patternMatchCpuFree(int* matchNum, int* matchIdx);
void patternMatchGpuFree(int* matchNumDev, int* matchIdxDev);
void printMatchOutputCpu(int* matchNum, int* matchIdx);
void printMatchOutputGpu(int* matchNumDev, int* matchIdxDev);