
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C"
{
#include "sgemm.h"
#include "cuda.h"
#include "utils.h"
#include "gemm.h"
}


__global__ void sgemm(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc)
	{
















		//printf("Kernel call has executed for gemm\n");
				
    		int row = blockDim.y * blockIdx.y + threadIdx.y;
    		int col = blockDim.x * blockIdx.x + threadIdx.x;
		//printf("A is %f\n",A_gpu[42]);			Will print multiple values as multiple threads

   		if (row > M || col > N) return; // Check for k as well, gotta use lda,ldb as well.
  
   			double prod = 0;
			int kk;
			for (kk = 0; kk < N; ++kk){
			    prod += A_gpu[row * lda + kk] * B_gpu[kk * ldb + col];
			    //printf("%d\n",prod);
			   }
			C_gpu[row*ldc + col] = ALPHA * prod + BETA * C_gpu[row*ldc+col];    
			printf("Kernel call has completed for gemm\n");		

	}


void sgemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc)
{
    //printf("Cublas has started Successfully\n");
    //printf("Printing out the parameters\n");
  //printf("Gpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);

    printf("These are the calls to gemm gpu\n");
    const dim3 blocksize(32,16);
    const dim3 gridsize(N/blocksize.y +1,M/blocksize.x+1);
    sgemm<<<gridsize,blocksize>>>(TA, TB, M, N, K, ALPHA, 
        A_gpu, lda, 
        B_gpu, ldb,
        BETA,
        C_gpu, ldc);


    check_error(cudaPeekAtLastError());
  //printf("Cublas has ended Successfully\n");
}




















