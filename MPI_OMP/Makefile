build:
	mpicc -fopenmp -c mpi_omp.c
	mpicc -lm -fopenmp -o mpiOmp  mpi_omp.o
clean:
	rm -f *.o ./mpiOmp