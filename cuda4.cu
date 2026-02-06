
// Addition von 2 arrays auf der GPU mit Hilfe von threads und blocks
// 
// compile:             nvcc cuda4.cu -o cuda4 && ./cuda4
// Aufruf mit Analyse:   nsys profile --stats=true -t cuda,nvtx ./cuda4

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


// kernel function
// this is run by multiple threads simultaneously
// each thread kernel is different !
// each thread has a variable threadIdx.x (thread index within the block)
// and a variable blockIdx.x (index of the block it belongs to)
// if we have enough threads each thread needs to do only one addition 
__global__ void add_gpu(const int *a, const int *b, int *c, int num_int) { 
	// each thread has a unique id
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	c[id] = a[id] + b[id];
}


int main() {
	print_device_info();

	int N = 10000;			// Anzahl integer im array 
	
	// Maximal 1024 threads pro block, empfohlen werden eher 512 für bessere Performance 
	int THREADS = 1000;		// Anzahl threads
	
	// maximal 2^32-1 blocks
	int BLOCKS = N / THREADS;		// wir brauchen 10 blocks
	
	size_t bytes = N * sizeof(int);		// wir brauchen N integer
	
	// Speicher auf dem host reservieren und initialisieren
	int *a = (int *) malloc(bytes);
	for (int i=0; i<N; i++)
		a[i] = 5;
	
	// Speicher auf dem host reservieren und initialisieren
	int *b = (int *) malloc(bytes);
	for (int i=0; i<N; i++)
		b[i] = 9;
	
	// Speicher auf dem host für das Ergebnis reservieren 
	int *c = (int *) malloc(bytes);
	
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
	add_gpu<<<BLOCKS, THREADS>>>(d_a, d_b, d_c, N);
	
	// Ergebnis nach c kopieren
	cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);
	
	printf("Ergebnis:\n");
	for (int i=0; i<10; i++)
		printf("%d ", c[i]);
	printf(" ... ");
	for (int i=N-10; i<N; i++)
		printf("%d ", c[i]);
	printf("\n");
	
	// Free memory on device
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}

