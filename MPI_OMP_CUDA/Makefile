build:
	mpicc -fopenmp -c mpi_omp_cuda.c  myProto.h 
	nvcc  -I /usr/local/cuda-9.1/samples/common/inc  -c cudaFunctions.cu -o cudaFunctions.o
	mpicc -fopenmp -o mpiCudaOpemMP  mpi_omp_cuda.o cudaFunctions.o /usr/local/cuda-9.1/lib64/libcudart_static.a -ldl -lrt -lstdc++
clean:
	rm -f *.o ./mpiCudaOpemMP *.h.gch

run:
	mpiexec -np 4 ./mpiCudaOpemMP
