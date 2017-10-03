// CUDA_1_PC.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <ctime>

using namespace std;

const unsigned int arrsize = 1000000;

void initArr(double* arr);
void sumArr(double* firstArr, double* secondArr);

int main()
{
	srand(time(0));

	unsigned int start_time = clock(); // начальное время

	double *a = new double[arrsize];
	double *b = new double[arrsize];

	initArr(a);
	initArr(b);

	sumArr(a, b);

	unsigned int end_time = clock(); // конечное время

	cout << "Time: " << end_time - start_time << endl;

	system("pause");

    return 0;
}

void initArr(double* arr) {
	for (int i = 0; i < arrsize; i++)
		arr[i] = rand() % 100 + 1;
	cout << "OK" << endl;
}

void sumArr(double* firstArr, double* secondArr) {
	for (int i = 0; i < arrsize; i++)
		firstArr[i] += secondArr[i];
	cout << "DONE" << endl;
}