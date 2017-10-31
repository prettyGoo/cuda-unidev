
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <ctime>
#include <iostream>

using namespace std;

cudaError_t mulWithCuda(float *A, float *B, float *C, int size);

__global__ void matrixMul(float *A, float *B, float *C, int size)
{
	unsigned __int16 i = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned __int16 j = threadIdx.y + blockIdx.y*blockDim.y;
	float sum = 0.0;

	for (unsigned __int16 k = 0; k < size; k++) {
		sum += A[k + i*size] * B[j + k*size];
	}

	C[j + i*size] = sum;
}

void initArray(float *arr, int size) {
	for (int i = 0; i < size; i++)
		arr[i] = rand() % 10;
}

int main()
{
	srand(time(0));
	const int arraySize = 4;
	float *A = new float[arraySize*arraySize];
	float *B = new float[arraySize*arraySize];
	float *C = new float[arraySize*arraySize];

	initArray(A, arraySize*arraySize);
	initArray(B, arraySize*arraySize);

	// Add vectors in parallel.
	cudaError_t cudaStatus = mulWithCuda(A, B, C, arraySize);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();

	for (int i = 0; i < arraySize*arraySize; i++) {
		cout << C[i];
	}

	delete A;
	delete B;
	delete C;

	system("pause");

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t mulWithCuda(float *A, float *B, float *C, int size)
{
	float *dev_a = 0;
	float *dev_b = 0;
	float *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, A, size * sizeof(int), cudaMemcpyHostToDevice);

	cudaStatus = cudaMemcpy(dev_b, B, size * sizeof(int), cudaMemcpyHostToDevice);

	// Launch a kernel on the GPU with one thread for each element.
	matrixMul << <1, size >> > (dev_a, dev_b, dev_c, size);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(C, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);


	return cudaStatus;
}
