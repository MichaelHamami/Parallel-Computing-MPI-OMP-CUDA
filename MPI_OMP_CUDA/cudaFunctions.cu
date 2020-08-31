#include <cuda_runtime.h>
#include <helper_cuda.h>
extern "C"{
#include "myProto.h"
}

__device__ int is_consevative(char seq1_char , char seq_other_char);
__device__ int is_semi_consevative(char seq1_char , char seq_other_char);

__global__ void alignment_score(double* array_score,int numElements,char *seq1,char *seq2,int offset,int hypen,double w1,double w2,double w3,double w4)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(numElements > i)
    {
		if(i < hypen)
		{
			if(seq1[i+offset] ==  seq2[i])
	        {
	            array_score[i] = w1;
	        }
	        // Check Consevative
	        else if(is_consevative(seq1[i+offset],seq2[i]))
	        {
	            array_score[i] = -w2;
	
	        }
	        // Check Semi Consevative
	        else if(is_semi_consevative(seq1[i+offset],seq2[i]))
	        {
	            array_score[i] = -w3;
	        }
	        else
	        {
	            array_score[i] = -w4;
	        }
		}
		else if(i > hypen)
		{
			if(seq1[i+offset] ==  seq2[i-1])
	        {
	            array_score[i] = w1;
	        }
	        // Check Consevative
	        else if(is_consevative(seq1[i+offset],seq2[i-1]))
	        {
	            array_score[i] = -w2;
	
	        }
	        // Check Semi Consevative
	        else if(is_semi_consevative(seq1[i+offset],seq2[i-1]))
	        {
	            array_score[i] = -w3;
	        }
	        else
	        {
	            array_score[i] = -w4;
	        }
		
		}
		else
		{
			array_score[i] = -w4;
		}
    }
}
extern "C"
void cuda_alignment_score(double* array_score,char *seq1,int len_seq1,char *seq2,int len_seq2,int offset,int hypen,double w1,double w2,double w3,double w4)
{
	// Error code to check return values for CUDA calls
     cudaError_t err = cudaSuccess;
     size_t size1 = len_seq1 * sizeof(char);
     size_t size2 = len_seq2 * sizeof(char);
     size_t size_score = (len_seq2+1) * sizeof(double);

    // Allocate memory on GPU to copy the data from the host
      char *d_seq1;
      char *d_seq2;
      double *d_score;
      err = cudaMalloc((void **)&d_seq1, size1);
    	if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
        }
      err = cudaMalloc((void **)&d_seq2, size2);
    	if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
        }
      err = cudaMalloc((void **)&d_score, size_score);
          	if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
        }
        
        
      // Copy data from host to the GPU memory
    err = cudaMemcpy(d_seq1, seq1, size1, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
        }
    err = cudaMemcpy(d_seq2, seq2, size2, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
        }
   	err = cudaMemcpy(d_score,array_score, size_score, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
        }
    
	// Calculates how many blocks to use based on the size of the arrays
    // Launch the Kernel
    int threadsPerBlock = 1024;
    int blocksPerGrid = (len_seq2 + threadsPerBlock - 1) / threadsPerBlock;
	alignment_score<<<blocksPerGrid, threadsPerBlock>>>(d_score,len_seq2+1,d_seq1,d_seq2,offset,hypen,w1,w2,w3,w4);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel -  %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
        }
    // Copy the  result from GPU to the host memory.
    err = cudaMemcpy(array_score, d_score, size_score, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy result array from device to host -%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Free allocated memory on GPU
    if (cudaFree(d_seq1) != cudaSuccess) {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
      if (cudaFree(d_seq2) != cudaSuccess) {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
          if (cudaFree(d_score) != cudaSuccess) {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
             
}
__device__ int is_consevative(char seq1_char , char seq_other_char)
{
	char conservative_groups[CONSERVATIVE_GROUP_NUMBER][CONSERVATIVE_GROUP_NUMBER_CHARS] = 
		{"NDEQ","NEQK","STA","MILV","QHRK","NHQK","FYW","HY","MILF"};
    int y;
    int x;
        for(y=0; y < CONSERVATIVE_GROUP_NUMBER;y++)
    {
        int is_char1 = 0;
        int is_char2 = 0;
        for(x=0;x<CONSERVATIVE_GROUP_NUMBER_CHARS;x++)
        {
            if(seq1_char == conservative_groups[y][x])
            {
                is_char1 = 1;
            }
            if(seq_other_char == conservative_groups[y][x])
            {
                is_char2 = 1;
            }
        }
       if (is_char1 && is_char2)
       {
           return 1;
       }
    }
    return 0;
}

__device__ int is_semi_consevative(char seq1_char , char seq_other_char)
{
	char semi_groups[SEMI_CONSERVATIVE_GROUP_NUMBER][SEMI_CONSERVATIVE_GROUP_NUMBER_CHARS] =
		{"SAG","ATV","CSA","SGND","STPA","STNK","NEQHRK","NDEQHK","SNDEQK","HFY","FVLIM"};
    int y;
    int x;
        for(y=0; y < SEMI_CONSERVATIVE_GROUP_NUMBER;y++)
    {
        int is_char1 = 0;
        int is_char2 = 0;
        for(x=0;x<SEMI_CONSERVATIVE_GROUP_NUMBER_CHARS;x++)
        {
            if(seq1_char == semi_groups[y][x])
            {
                is_char1 = 1;
            }
            if(seq_other_char == semi_groups[y][x])
            {
                is_char2 = 1;
            }
        }
       if (is_char1 && is_char2)
       {
           return 1;
       }

    }
    return 0;
}

