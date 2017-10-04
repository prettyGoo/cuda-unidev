#include <iostream>
#include <ctime>
#include <cstdio>

using namespace std;

const unsigned int arrsize = 100000000;

void initArr(float* arr);
void sumArr(float* firstArr, float* secondArr, float* finalArr);

int main()
{
	srand(time(0));

	unsigned int start_time = clock();

	float *a = new float[arrsize];
	float *b = new float[arrsize];
	float *c = new float[arrsize];

	initArr(a);
	initArr(b);

	sumArr(a, b, c);

	unsigned int end_time = clock();

	cout << "Time: " << end_time - start_time << endl;

	delete a;
	delete b;
	delete c;

	system("pause");
	return 0;
}

void initArr(float* arr) {
	for (int i = 0; i < arrsize; i++)
		arr[i] = rand() % 100 + 1;
	cout << "OK" << endl;
}

void sumArr(float* firstArr, float* secondArr, float* finalArr) {
	for (int i = 0; i < arrsize; i++)
		finalArr[i] = firstArr[i] + secondArr[i];
	cout << "DONE" << endl;
}
