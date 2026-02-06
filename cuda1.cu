
// einfache Addition auf der GPU
// 
// compile:             nvcc cuda1.cu -o cuda1 && ./cuda1
// Aufruf mit Analyse:   nsys profile --stats=true -t cuda,nvtx ./cuda1

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


__global__ void add_gpu(const int *a, const int *b, int *c) { 
	c[0] = a[0] + b[0];
}


int main() {
	print_device_info();
	int bytes = 1 * sizeof(int);		// wir brauchen nur ein INT
	
	int a = 5, b = 9, c;
	
	int *d_a;					// pointer to integer auf dem Device
	cudaMalloc(&d_a, bytes);	// ein INT reservieren auf der GPU
	
	int *d_b;					// pointer to integer auf dem Device
	cudaMalloc(&d_b, bytes);	// ein INT reservieren auf der GPU
	
	int *d_c;					// pointer to integer auf dem Device
	cudaMalloc(&d_c, bytes);	// ein INT reservieren auf der GPU
	
	// daten von Host zu Device kopieren
	cudaMemcpy(d_a, &a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, bytes, cudaMemcpyHostToDevice);
	
	// kernel aufrufen
	add_gpu<<<1, 1>>>(d_a, d_b, d_c);
	
	// Ergebnis nach c kopieren
	cudaMemcpy(&c, d_c, bytes, cudaMemcpyDeviceToHost);

	printf("Ergebnis: %d\n\n", c);
	
	// Free memory on device
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	
	return 0;
}

