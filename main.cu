
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <stdio.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <ctime>

using namespace std;

void sum_vectors();
void gpu_sum(float *a, float *b, float *c, int arary_size);

int main()
{
	srand(time(0));
	unsigned int start_time = clock();

	sum_vectors();

	unsigned int end_time = clock();
	cout << "Time: " << end_time - start_time << endl;

	system("pause");
	return 0;
}


// ====================================
__global__ void sum_kernel(float *a, float *b, float *c) {
	int idx = threadIdx.x + threadIdx.y + threadIdx.z;

	c[idx] = a[idx] + b[idx];
}

void sum_vectors() {

	int array_size = 1000000;
	float *a = new float[array_size];
	float *b = new float[array_size];
	float *c = new float[array_size];

	for (int i = 0; i < array_size; i++) {
		a[i] = rand() % 10;
		b[i] = rand() % 10;
		c[i] = 0;
	}

	gpu_sum(a, b, c, array_size);
	
	delete a;
	delete b;
	delete c;

	return;
}

void gpu_sum(float *a, float *b, float *c, int arary_size) {
	int raw_size = arary_size * sizeof(float);

	float *aDevice = NULL;
	float *bDevice = NULL;
	float *cDevice = NULL;

	cudaError_t cuerr;
	cuerr = cudaMalloc((void**)&aDevice, raw_size);
	if (cuerr != cudaSuccess) {
		fprintf(stderr, "Cannot allocate memory for aDevice %s\n", cudaGetErrorString(cuerr));
		system("pause");
	}

	cuerr = cudaMalloc((void**)&bDevice, raw_size);
	if (cuerr != cudaSuccess) {
		fprintf(stderr, "Cannot allocate memory for bDevice %s\n", cudaGetErrorString(cuerr));
		system("pause");
	}

	cuerr = cudaMalloc((void**)&cDevice, raw_size);
	if (cuerr != cudaSuccess) {
		fprintf(stderr, "Cannot allocate memory for cDevice %s\n", cudaGetErrorString(cuerr));
		system("pause");
	}

	cuerr = cudaMemcpy(aDevice, a, raw_size, cudaMemcpyHostToDevice);
	if (cuerr != cudaSuccess) {
		fprintf(stderr, "Cannot copy data from a to aDevice: %s\n", cudaGetErrorString(cuerr));
		system("pause");
	}

	cuerr = cudaMemcpy(bDevice, b, raw_size, cudaMemcpyHostToDevice);
	if (cuerr != cudaSuccess) {
		fprintf(stderr, "Cannot copy data from b to bDevice: %s\n", cudaGetErrorString(cuerr));
		exit(EXIT_FAILURE);
	}

	sum_kernel<<<10000, 10 >>>(aDevice, bDevice, cDevice);
	cuerr = cudaGetLastError();
	if (cuerr != cudaSuccess) {
		fprintf(stderr, "Cannot launch CUDA kernel: ^%sn", cudaGetErrorString(cuerr));
		system("pause");
	}

	cuerr = cudaDeviceSynchronize();
	if (cuerr != cudaSuccess) {
		fprintf(stderr, "Cannot synch CUDA kernel: ^%sn", cudaGetErrorString(cuerr));
		system("pause");
	}

	cuerr = cudaMemcpy(c, cDevice, raw_size, cudaMemcpyDeviceToHost);
	if (cuerr != cudaSuccess) {
		fprintf(stderr, "cannot copy data from GPU to CPU: %s\n", cudaGetErrorString(cuerr));
		system("pause");
	}

	cudaFree(aDevice);
	cudaFree(bDevice);
	cudaFree(cDevice);

	return;
}
