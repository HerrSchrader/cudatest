
// Multiplikation von 2 Matrizen auf der GPU mit Hilfe von threads und blocks
// 
// compile:             nvcc cuda6.cu -o cuda6 && ./cuda6
// Aufruf mit Analyse:   nsys profile --stats=true -t cuda,nvtx ./cuda6

using namespace std;                                     // Compileranweisung für C++ Code

#include <stdio.h>			// für printf

// you must first call the cudaGetDeviceProperties() function, then pass 
// the devProp structure returned to this function:
int getSPcores(cudaDeviceProp devProp)
{  
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major){
     case 2: // Fermi
      if (devProp.minor == 1) cores = mp * 48;
      else cores = mp * 32;
      break;
     case 3: // Kepler
      cores = mp * 192;
      break;
     case 5: // Maxwell
      cores = mp * 128;
      break;
     case 6: // Pascal
      if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
      else if (devProp.minor == 0) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
     case 7: // Volta and Turing
      if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
     case 8: // Ampere
      if (devProp.minor == 0) cores = mp * 64;
      else if (devProp.minor == 6) cores = mp * 128;
      else if (devProp.minor == 9) cores = mp * 128; // ada lovelace
      else printf("Unknown device type\n");
      break;
     case 9: // Hopper
      if (devProp.minor == 0) cores = mp * 128;
      else printf("Unknown device type\n");
      break;
     case 10: // Blackwell
      if (devProp.minor == 0) cores = mp * 128;
      else printf("Unknown device type\n");
      break;
     case 12: // Blackwell
      if (devProp.minor == 0) cores = mp * 128;
      else printf("Unknown device type\n");
      break;
     default:
      printf("Unknown device type\n"); 
      break;
      }
    return cores;
}

void print_device_info(){
	int nDevices;
	cudaGetDeviceCount(&nDevices);

	printf("Number of devices: %d\n", nDevices);

	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (MHz): %d\n",
			prop.memoryClockRate/1024);
		printf("  Memory Bus Width (bits): %d\n",
			prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %.1f\n",
			2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
		printf("  Total global memory (Gbytes) %.1f\n",(float)(prop.totalGlobalMem)/1024.0/1024.0/1024.0);
		printf("  Shared memory per block (Kbytes) %.1f\n",(float)(prop.sharedMemPerBlock)/1024.0);
		printf("  minor-major: %d-%d\n", prop.minor, prop.major);
		printf("  Number of multiprocessors: %d\n", prop.multiProcessorCount);
		printf("  Number of cores: %d\n", getSPcores(prop));
		printf("  Warp-size: %d\n", prop.warpSize);
		printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
		printf("  Concurrent computation/communication: %s\n\n",prop.deviceOverlap ? "yes" : "no");
	}
}

void print_result(int *c, int N){
		
	printf("Ergebnis:\n");
	for (int i=0; i<3; i++){
		for (int j=0; j<10; j++)
			printf("%d ", c[i*N+j]);
		printf(" ... ");
		for (int j=N-10; j<N; j++)
			printf("%d ", c[i*N+j]);
		printf("\n");
	};
	printf("...\n");
	
	for (int i=N-3; i<N; i++){
		for (int j=0; j<10; j++)
			printf("%d ", c[i*N+j]);
		printf(" ... ");
		for (int j=N-10; j<N; j++)
			printf("%d ", c[i*N+j]);
		printf("\n");
	};
	printf("\n");
}



// kernel function
// this is run by multiple threads simultaneously
// each thread kernel is different !
// each thread has a variable threadIdx.x (thread index within the block)
// and a variable blockIdx.x (index of the block it belongs to)

// blockIdx.x ist die Zeile
// threadIdx.x ist die Spalte
// jeder thread berechnet genau ein Ergebnis
__global__ void mul_gpu(const int *a, const int *b, int *c, int N) { 
	// each thread has a unique id
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.x;
	int j = threadIdx.x;
	int sum = 0;
	for (int k=0; k<N; k++){
		sum += a[i*N + k] * b[k * N + j];
	};
	c[id] = sum; 
}


void host_mul(int *a, int *b, int *c, int N){
	for (int i=0; i<N; i++){					// Zeile i
		for (int j=0; j<N; j++){				// Spalte j
			int sum = 0;
			for (int k=0; k<N; k++)
				sum += a[i*N +k] * b[k*N +j];
			c[i*N+j] = sum;
		}
	}
}

int main() {
	print_device_info();

	int N = 1000;			// Anzahl Spalten und Anzahl Zeilen im array 

	size_t bytes = N * N * sizeof(int);		// wir brauchen N*N integer pro matrix
	
	// Speicher auf dem host reservieren und initialisieren
	int *a = (int *) malloc(bytes);
	for (int i=0; i<N; i++)
		for (int j=0; j<N; j++)
			a[i*N+j] = 5;
	
	// Speicher auf dem host reservieren und initialisieren
	int *b = (int *) malloc(bytes);
	for (int i=0; i<N; i++)
		for (int j=0; j<N; j++)
			b[i*N+j] = 9;
	
	// Speicher auf dem host für das Ergebnis reservieren 
	int *c = (int *) malloc(bytes);
	
	// Berechnung auf dem host
	host_mul(a, b, c, N);
	print_result(c, N);
		
	
	
	// Maximal 1024 threads pro block, empfohlen werden eher 512 für bessere Performance 
	int THREADS = N;			// Anzahl threads = Anzahl Zeilen
	
	// maximal 2^32-1 blocks
	int BLOCKS = N;				// wir brauchen N blocks
		
	int *d_a;					// pointer to integer auf dem Device
	cudaMalloc(&d_a, bytes);	// N * INT reservieren auf der GPU
	
	int *d_b;					// pointer to integer auf dem Device
	cudaMalloc(&d_b, bytes);	// N * INT reservieren auf der GPU
	
	int *d_c;					// pointer to integer auf dem Device
	cudaMalloc(&d_c, bytes);	// N * INT reservieren auf der GPU
	
	// daten von Host zu Device kopieren
	cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);
	
	// kernel aufrufen
	mul_gpu<<<BLOCKS, THREADS>>>(d_a, d_b, d_c, N);
	
	// Ergebnis nach c kopieren
	cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);
	
	print_result(c, N);
		
	// Free memory on device
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}

