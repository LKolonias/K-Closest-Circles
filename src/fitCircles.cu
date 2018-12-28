#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include "magma.h"
#include "magma_lapack.h"
#include <sys/time.h>
#include<curand.h>
#include<curand_kernel.h>


// Function to measure the time in miliseconds
double time_diff(struct timeval x , struct timeval y)
{
    double x_ms , y_ms , diff;   
    x_ms = (double)x.tv_sec*1000000 + (double)x.tv_usec;
    y_ms = (double)y.tv_sec*1000000 + (double)y.tv_usec; 
    diff = (double)y_ms - (double)x_ms;
     
    return diff;
}

//Function to handle cuda errors
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

//Needed for magma_dgels
# define max(a,b) (((a)<(b ))?( b):(a))



// Device function which calculates the vector and constructs the matrix
// which participate in magma_dgels function.
__global__ void vecprod(double *x, double *y,double *b, double *A, int numofpoints){
        double h1,h2;
        int block = threadIdx.x;
        if(blockIdx.x==0){
                if(block<numofpoints){
                        h1=x[block]*x[block];
                        h2=y[block]*y[block];
                        b[block]=-(h1+h2);
                }
        }
        if(blockIdx.x==1){
                if(block<numofpoints){
                        A[block] = x[block];
                        A[block+numofpoints] = y[block];
                        A[block+2*numofpoints] = 1;
                }
        }
}

// Fitring function calculates the center and radius of the circles
void fitring(double pts[50][2], double results[], int l){
        int j;
        double x[l],y[l],b[l],A[l*3];
        double *dev_x,*dev_y,*dev_b, *dev_A;

        for(j=0; j<l; j++){
                x[j]=pts[j][0];
                y[j]=pts[j][1];

        }
	//allocating memory in device, copying x and y, and calling function vecprod to generate matrix and vector for magma_dgels
        HANDLE_ERROR( cudaMalloc( (void**)&dev_x, l * sizeof(double) ) );
        HANDLE_ERROR( cudaMalloc( (void**)&dev_y, l * sizeof(double) ) );
        HANDLE_ERROR( cudaMalloc( (void**)&dev_b, l * sizeof(double) ) );
        HANDLE_ERROR( cudaMalloc( (void**)&dev_A, l * 3 * sizeof(double) ) );

		HANDLE_ERROR(cudaMemcpy(dev_x, x, l * sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(dev_y, y, l * sizeof(double), cudaMemcpyHostToDevice));

        vecprod<<<2,l>>>(dev_x, dev_y, dev_b,dev_A, l);

        HANDLE_ERROR(cudaMemcpy(b, dev_b, l * sizeof(double),cudaMemcpyDeviceToHost));

        HANDLE_ERROR(cudaMemcpy(A, dev_A, l * 3 * sizeof(double),cudaMemcpyDeviceToHost));


cudaFree(dev_x);
cudaFree(dev_y);
cudaFree(dev_A);


//Setting up magma parameters
double a[3];
double *mag_A, *mag_b;
magma_init();

magma_int_t m,n;
m=l;
n=3;

magma_int_t info, nb, lworkgpu,l1,l2,lhwork;
double *hwork, tmp[1];

nb = magma_get_dgeqrf_nb (m); 
lworkgpu = (m-n + nb )*(1 +2* nb);
//Allocating memory on device for matrix and vector
magma_dmalloc (&mag_b , m*1); 
magma_dmalloc (&mag_A , m*n);

l1 = ( magma_int_t ) MAGMA_D_REAL ( tmp [0] );
l2 = ( magma_int_t ) MAGMA_D_REAL ( tmp [0] );
lhwork = max ( max ( l1 , l2 ), lworkgpu );

magma_dmalloc_cpu (& hwork , lhwork );

//Setting matrices in device for magma_dgels
magma_dsetmatrix ( m, n, A, m, mag_A , m ); 
magma_dsetmatrix ( m, 1 , b, m, mag_b , m );

//magma_dgels solving the least squares problem
magma_dgels_gpu(MagmaNoTrans, m, n, 1,mag_A, m, mag_b, m, hwork, lworkgpu, &info);

//Getting matrix from device and returning results
magma_dgetmatrix ( n, 1 , mag_b , m , b, n );

results[0] = -0.5* b[0];
results[1] = -0.5* b[1];
results[2] = sqrt((b[0]*b[0]+b[1]*b[1])/4-b[2]);

}


//Fitcircles function fits circles to points
void fitCircles(double points[][2],double point[], int i, double circles[][3],int numofpoints){
	int j,k,l,N;
	N = numofpoints;
	double pts[numofpoints][2];

	// Getting the points which belong to each circle
	for(j=0; j<i; j++){
		l=0;
		for(k=0; k<N; k++){
			if(point[k]==j){
				pts[l][0] = points[k][0];
				pts[l][1] = points[k][1];
				l++;
			}
		}
		//if the number of points is less than 4 no circle is created
		if(l<4){
			circles[j][0]=0;
			circles[j][1]=0;
			circles[j][2]=100;
		}
		else {
			double res[3];
			//calling fitring to calculate the circle due to the points
			fitring(pts,res,l);
			circles[j][0] = res[0];
			circles[j][1] = res[1];
			circles[j][2] = res[2];
		}
	}
}

//circle Dist function calculates the distance of the points from each circle. Called only from device
__device__ void circleDist(double po1, double po2, double *circles1, int i, float d[]){
	
	double xa[5],yb[5],r[5],h1;
	int j;
	//Calculating distance from each circle
	for(j=0; j<i; j++){
		xa[j] = po1 - circles1[j*3];
		yb[j] = po2 - circles1[j*3 + 1];
		r[j] = circles1[j*3 + 2];
		h1 = xa[j]*xa[j] + yb[j]*yb[j] - r[j]*r[j];
		d[j] = h1*h1;	
	}
}

//findPoints finds the points of a circle by calling circleDist and assigning each point to its nearest circle
__global__ void findPoints(double *points, double *circles1, double *points1, int numofpoints, int i, int *res){
	
	int block = blockIdx.x,j,pos;
	float d[5], min;

	circleDist(points[2*block],points[2*block+1],circles1,i,d);
	
	min = d[0];
	pos = 0;	
	for(j=0; j<i; j++){
		if(d[j]<min){
			min = d[j];
			pos = j;
		}
	}
	//Using atomicAdd to calculate the number of changes between the threads
	if (points1[block]!=pos){
		atomicAdd(&res[0], 1);
	}
	points1[block] = pos;
}

//Randperm function used for generating random points
void randperm(int n, int perm[])
{
        int i, j, t;

        for(i=0; i<n; i++)
                perm[i] = i;
        for(i=0; i<n; i++) {
                j = rand()%(n-i)+i;
                t = perm[j];
                perm[j] = perm[i];
                perm[i] = t;
        }
}


//Setting up kernel for generating random numbers
__global__ void setup_kernel ( curandState * state, unsigned long seed )
{
        int id = threadIdx.x;
        curand_init ( seed, id, 0, &state[id] );

}

//Generating random numbers uniformly distributed in [-1 1] in the device using curand's functions
__global__ void random(double *N, curandState* globalState)
{
        curandState localState = globalState[threadIdx.x];
        float random = curand_uniform(&localState);
        globalState[threadIdx.x] = localState;
       
	//Half of the numbers will be negative
	if((threadIdx.x % 2)==0){
                N[threadIdx.x] = random;
        }else{
                N[threadIdx.x] = -random;
        }
}


//ClosestCircle function calculates the circles and the points belonging to them after calling recurcively findPoints and fitCircles functions
void closestCircles(double points[][2], int i, int maxIter, int initializeCirclesFirst, int numofevent, int numofpoints,double circles1[][3], double points1[]){
		
	int j,k,N,numChanges,u;
	N = numofpoints;
	
	//In first attempt generate random circles
	if(initializeCirclesFirst==1){
	
	curandState* devStates;
	HANDLE_ERROR( cudaMalloc( (void**)&devStates, 3 * 5 * sizeof(curandState) ) );
	
	setup_kernel<<<1,15>>>(devStates,unsigned(time(NULL)));

	double *dev_c1;
	HANDLE_ERROR( cudaMalloc( (void**)&dev_c1, 3 * 5 * sizeof(double) ) );
	random<<<1,15>>>(dev_c1,devStates);
	HANDLE_ERROR(cudaMemcpy(circles1, dev_c1, 3 * 5 * sizeof(double),cudaMemcpyDeviceToHost));
		
		numChanges = 1;
		for(j=0; j<numofpoints; j++){
			points1[j]=0;			
		}
		
	}
	//in second attempt generate random points
	else{
		int idx[N];
		randperm(N,idx);
		int cIdx = 0;
		
		for(k=0; k<N; k++){
			u=idx[k];
			points1[u] = cIdx;
			cIdx = cIdx+1;
			if(cIdx > i-1){
				cIdx = 0;
			}		
		}
		
		fitCircles(points, points1, i, circles1, numofpoints);	
	}


	numChanges = 1;

	while ((numChanges >0) && (maxIter>0)){
		
		//Setting up memory and parameters to call findPoints in device
		int res[1], *dev_res;
		res[0] = 0;
		double *dev_points, *dev_circles1, *dev_points1 ;	
		HANDLE_ERROR( cudaMalloc( (void**)&dev_points, numofpoints * 2 * sizeof(double) ) );
		HANDLE_ERROR( cudaMalloc( (void**)&dev_circles1, 5 * 3 * sizeof(double) ) );
		HANDLE_ERROR( cudaMalloc( (void**)&dev_points1, numofpoints * sizeof(double) ) );
		HANDLE_ERROR( cudaMalloc( (void**)&dev_res, 1 * sizeof(int) ) );

		HANDLE_ERROR(cudaMemcpy(dev_points, points, numofpoints * 2  * sizeof(double), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(dev_circles1, circles1, 5 * 3 * sizeof(double), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(dev_points1, points1, numofpoints * sizeof(double), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(dev_res, res, 1 *sizeof(int), cudaMemcpyHostToDevice));

		findPoints<<<N,1>>>(dev_points, dev_circles1, dev_points1, N, i, dev_res);
		HANDLE_ERROR(cudaMemcpy(points1, dev_points1, numofpoints * sizeof(double),cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(res, dev_res, 1 * sizeof(int),cudaMemcpyDeviceToHost));

		numChanges = res[0];
		maxIter = maxIter - 1;
		
		cudaFree(dev_points);
		cudaFree(dev_circles1);
		cudaFree(dev_points1);
		cudaFree(dev_res);
		fitCircles(points, points1, i, circles1, numofpoints);	
		
	}

}

//pruneCircles removes all circles which don't fit the criteria of minimum number of points
void pruneCircles(double circles[][3],double points[], float radiusThreshold, int i, int numofpoints, int size[]){
	
	int j,k,l,w,h1;
	double prunedC[5][3], prunedP[numofpoints], locations[numofpoints];
	w=0;
	for(j=0; j<numofpoints; j++){
		prunedP[j] = 0;
	}

	for(j=0; j<i; j++){
		l=0;	
		for(k=0; k<numofpoints; k++){
			if(points[k]==j){
				locations[l] = k;
				l++;			
			}
		}
		if(circles[j][2]<radiusThreshold){ continue; }
		if(l<4){ continue; }
		for(k=0; k<l; k++){
			h1 = locations[k];
			prunedP[h1] =j;
		}
		prunedC[w][0] = circles[j][0];
		prunedC[w][1] = circles[j][1];
		prunedC[w][2] = circles[j][2];
		w++;
	}
	size[0]=w+1;
	for(j=0; j<w; j++){
		circles[j][0] = prunedC[j][0];
		circles[j][1] = prunedC[j][1];
		circles[j][2] = prunedC[j][2];
	}
	for(j=0; j<numofpoints; j++){
		points[j] = prunedP[j];
	}
}


//Calculates LAD and adding an overfit penalty in the error.
__global__ void circleFitError(double *points,  double *circles1, float overfitPenalty,int i, int size, int numofpoints, float *err){


	int block=blockIdx.x, j;
	float d[5], min;
	float h1;

	circleDist(points[2*block],points[2*block+1],circles1,size,d);
	
	min = d[0];
	for(j=0; j<i; j++){
		if(d[j]<min){
			min = d[j];
		}
	}
	atomicAdd(&err[0],min);
	

	if(block==0){
		h1 = overfitPenalty * i * i;
		atomicAdd(&err[0],h1);
	}	
}


//KCC algorithm calling all functions.
void kcc(double points[][2], int i, int maxIter, float radiusThreshold, float overfitPenalty, int numofevent, int numofpoints, double cir[][3], float err[],int s[]){
	int size1[1],size2[1],j,u;
	double circles1[5][3],circles2[5][3];
	double points1[numofpoints],points2[numofpoints];
	
	float err1[1],err2[1];
	double *dev_points, *dev_points1, *dev_circles1, *dev_points2, *dev_circles2;
	float *dev_err1, *dev_err2;	
	err1[0]=err2[0]=0.0;


	closestCircles(points, i, maxIter, 1, numofevent, numofpoints,circles1,points1);

	pruneCircles(circles1,points1,radiusThreshold,i,numofpoints,size1);

	HANDLE_ERROR( cudaMalloc( (void**)&dev_points, numofpoints * 2 * sizeof(double) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_circles1, 5 * 3 * sizeof(double) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_err1, 1 * sizeof(float) ) );
	

	HANDLE_ERROR(cudaMemcpy(dev_points, points, numofpoints * 2  * sizeof(double), cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMemcpy(dev_circles1, circles1, 5 * 3 * sizeof(double), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_err1, err1, 1 * sizeof(float), cudaMemcpyHostToDevice));


	circleFitError<<<numofpoints,1>>>(dev_points,dev_circles1,overfitPenalty,i,size1[0],numofpoints,dev_err1);


	HANDLE_ERROR(cudaMemcpy(err1, dev_err1, 1 * sizeof(float), cudaMemcpyDeviceToHost));

	cudaFree(dev_points1);
	cudaFree(dev_circles1);
	cudaFree(dev_err1);

//Second attempt in which random points are generated first
	closestCircles(points, i, maxIter, 0, numofevent, numofpoints,circles2,points2);
	pruneCircles(circles2,points2,radiusThreshold,i,numofpoints,size2);

	HANDLE_ERROR( cudaMalloc( (void**)&dev_circles2, 5 * 3 * sizeof(double) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_err2, 1 * sizeof(float) ) );
	

	HANDLE_ERROR(cudaMemcpy(dev_circles2, circles2, 5 * 3 * sizeof(double), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_err2, err2, 1 * sizeof(float), cudaMemcpyHostToDevice));


	circleFitError<<<numofpoints,1>>>(dev_points,dev_circles2,overfitPenalty,i,size2[0],numofpoints,dev_err2);


	HANDLE_ERROR(cudaMemcpy(err2, dev_err2, 1 * sizeof(float), cudaMemcpyDeviceToHost));
	cudaFree(dev_points);
	cudaFree(dev_points2);
	cudaFree(dev_circles2);
	cudaFree(dev_err2);

	int q;
	//decide which set is returned by comparing errors
	if(err1[0]<=err2[0]){
		q=1;
		for(j=0; j<size1[0]; j++){
			cir[j][0] = circles1[j][0];
			cir[j][1] = circles1[j][1];
			cir[j][2] = circles1[j][2];
		}
		err[0] = err1[0];
		s[0] - size1[0];
	}else{
		q=0;
		for(j=0; j<size2[0]; j++){
			cir[j][0] = circles2[j][0];
			cir[j][1] = circles2[j][1];
			cir[j][2] = circles2[j][2];
		}
		err[0] = err2[0];
		s[0] = size2[0];
	}
	
}




int main( int argc, char* argv[] ) {
	
	FILE* file = fopen("batch00.dat", "r");
	char line[128];
	int numofevents,i,j,k;
	struct timeval tv[2];

	fgets(line,sizeof(line),file);
	numofevents = atoi(line);
	
	int numofpoints[numofevents];
	double events[numofevents];

	gettimeofday (&tv[0], NULL);
		
	//for loop to reed progressively points from file and calling kcc 4 times for each event
	for (i=0; i<numofevents; i++){
		
		
		fgets(line,sizeof(line),file);
		numofpoints[i] = atoi(line);	
		double points[numofpoints[i]][2];
		
		for (j=0; j<numofpoints[i]; j++){
			fgets(line,sizeof(line),file);
			
			char* buffer;
			buffer = strtok(line," ");
			
			int z = 0;
			while(buffer){
				
				points[j][z] = atof(buffer);
				buffer = strtok(NULL," ");
				z++;
			}
			
		}
			
		int maxK = 5;
		int minK = 2;
		int Ks[maxK-minK+1];
		
		for (k=minK; k<=maxK; k++ ){
				Ks[k-minK] = k;
		}
		
		float radiusThreshold = 0.1;
		int maxIter = 100;
		float overfitPenalty = 0.001;
		
		double circles[5][3];
		double error;
		int K,size;
		for (k=0; k<4; k++){
			double cir[5][3];
			float err[1];
			int s[1];
				
			kcc(points,Ks[k],maxIter, radiusThreshold, overfitPenalty, i, numofpoints[i],cir,err,s);		
	
	//deciding which circle set to keep by comparing errors
	if(k==0){
		for(j=0; j<s[0]; j++){
			circles[j][0] = cir[j][0];
			circles[j][1] = cir[j][1];
			circles[j][2] = cir[j][2];
		}
		error = abs(err[0]);
		size = s[0]; K=Ks[k];
	}else{
		if(abs(err[0])<abs(error)){
			
			for(j=0; j<s[0]; j++){
				circles[j][0] = cir[j][0];
				circles[j][1] = cir[j][1];
				circles[j][2] = cir[j][2];
			}
			error = abs(err[0]);
			size = s[0]; K=Ks[k];
		}
	}
	
		}
		//Printing results
		printf("\n\nEvent: %d \n",i);
		for (k=0; k<size; k++){
			printf("%.8f  %.8f  %.8f \n",circles[k][0],circles[k][1], circles[k][2]);
		}
		float LAD;
		LAD = error - overfitPenalty *(float)K*(float)K;
		printf("LAD: %.12f",LAD);
	}	

	
	//Printing time 
	gettimeofday (&tv[1], NULL);
	printf("\n\nTime elapsed: %.0lf microsec\n\n", time_diff(tv[0],tv[1]) );
	return 0;
}
