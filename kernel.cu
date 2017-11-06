#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <ctime>
#include <stdio.h>
#include <iostream>

using namespace std;

const unsigned int arrsize = 160;
const int BLOCK_SIZE = 16;

void initArr(double* arr, bool fill_with_zero);
void multiplyMatrixes(double* firstArr, double* secondArr, double* finalArr);
double sumArrayElems(double *arr);


//=================== GPU ===================
__global__ void matrixmul_kernel(double *A, double *B, double *C) {
	double sum = 0;

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	for (int k = 0; k < arrsize; k++) {
		sum += A[row * arrsize + k] * B[k * arrsize + col];
	}

	C[row * arrsize + col] = sum;
}


int main()
{
	srand(time(0));

	double *A = new double[arrsize*arrsize];
	double *B = new double[arrsize*arrsize];
	double *C = new double[arrsize*arrsize];

	initArr(A, false);
	initArr(B, false);
	initArr(C, true);

	printf("The size of elems in A and B is %d x %d", arrsize, arrsize);
	//=================== CPU //===================
	cout << endl << "CPU" << endl;

	clock_t start_time = clock();
	multiplyMatrixes(A, B, C);
	clock_t end_time = clock();

	cout << "Time: " << end_time - start_time << endl;
	cout << "Sum of the elements after CPU concat: " << sumArrayElems(C) << endl;

	//=================== GPU ===================
	cout << endl << "GPU" << endl;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	size_t raw_size = arrsize * arrsize * sizeof(double);

	double *aDevice = nullptr;
	double *bDevice = nullptr;
	double *cDevice = nullptr;

	// Выделить память под 
	cudaMalloc((void**)&aDevice, raw_size);
	cudaMalloc((void**)&bDevice, raw_size);
	cudaMalloc((void**)&cDevice, raw_size);

	// Копировать массивы A и B в память GPU
	cudaMemcpy(aDevice, A, raw_size, cudaMemcpyHostToDevice);
	cudaMemcpy(bDevice, B, raw_size, cudaMemcpyHostToDevice);

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((arrsize + dimBlock.x - 1) / dimBlock.x, (arrsize + dimBlock.y - 1) / dimBlock.y);

	cudaEventRecord(start);
	matrixmul_kernel <<<dimGrid, dimBlock>>> (aDevice, bDevice, cDevice);
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);

	cudaDeviceSynchronize();
	cudaMemcpy(C, cDevice, raw_size, cudaMemcpyDeviceToHost);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cout << "Time: " << milliseconds << endl;
	cout << "Sum of the elements after GPU concat: " << sumArrayElems(C) << endl;

	// ==========================================
	cudaFree(aDevice);
	cudaFree(bDevice);
	cudaFree(cDevice);

	delete A;
	delete B;
	delete C;

	system("pause");
	return 0;
}

//=================== CPU ===================
// Функция, которая принимает на вход массив и инициализирует его рандомными значениями
void initArr(double* arr, bool fill_with_zero) {
	if (fill_with_zero) {
		for (int i = 0; i < arrsize*arrsize; i++)
			arr[i] = 0;
	}
	else {
		for (int i = 0; i < arrsize*arrsize; i++)
			arr[i] = 2;// rand() % 10;
	}

}

// Функция, которая складывает поэлементно два массива
void multiplyMatrixes(double* firstArr, double* secondArr, double* finalArr) {

	double row_sum = 0;
	double column_sum = 0;
	double sum = 0;

	int final_array_index = 0;

	// обходим каждую строку первой матрицы
	for (int i = 0; i < arrsize; i++) {
		// обходим каждый столбец второй матрицы
		for (int j = 0; j < arrsize; j++) {
			for (int k = 0; k < arrsize; k++) {
				sum += firstArr[i*k] * secondArr[j*k];
			}

			finalArr[final_array_index] = sum;
			sum = 0;
			final_array_index++;
		}
	}
}

// Функция, которая находит сумму всех элементов в массиве
double sumArrayElems(double* arr)
{
	double sum = 0;

	for (int i = 0; i < arrsize*arrsize; i++)
		sum += arr[i];
	return sum;
}
