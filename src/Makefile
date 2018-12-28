NVCC = nvcc

FLAGS = -arch=sm_20 -O3

FLAGS1 = -DADD_ -I/opt/magma/1.7.0/openblas/gcc/include -I/usr/local/cuda/include -w -c

FLAGS2 = -L/opt/magma/1.7.0/openblas/gcc/lib -L/usr/local/cuda/lib64 -L/opt/openblas/0.2.15/gcc/lib -lmagma -lcublas -lcudart -lopenblas

all: fitCircles.o hpc.o


hpc.o: fitCircles.o
	$(NVCC) $(FLAGS) -o hpc fitCircles.o  $(FLAGS2)
        
fitCircles.o:
	$(NVCC) $(FLAGS) $(FLAGS1) fitCircles.cu
