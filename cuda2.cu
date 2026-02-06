
// Addition von 2 arrays auf der GPU
// 
// compile:             nvcc cuda2.cu -o cuda2 && ./cuda2
// Aufruf mit Analyse:   nsys profile --stats=true -t cuda,nvtx ./cuda2

using namespace std;                                     // Compileranweisung für C++ Code

#include <stdio.h>			// für printf

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
		printf("  Warp-size: %d\n", prop.warpSize);
		printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
		printf("  Concurrent computation/communication: %s\n\n",prop.deviceOverlap ? "yes" : "no");
	}
}


__global__ void add_gpu(const int *a, const int *b, int *c, int num_int) { 
	for (int i=0; i<num_int; i++)
		c[i] = a[i] + b[i];
}


int main() {
	print_device_info();

	int N = 1000;			// Anzahl integer im array 
	
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
	add_gpu<<<1, 1>>>(d_a, d_b, d_c, N);
	
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

