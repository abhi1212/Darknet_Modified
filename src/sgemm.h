#ifndef SGEMM_H
#define SGEMM_H
#ifdef GPU
#include "darknet.h"

void sgemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc);

#endif
#endif
