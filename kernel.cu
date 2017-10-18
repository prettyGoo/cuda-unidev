#include <stdio.h>
#include <cuda_runtime_api.h>
#include <iostream>

int getNumberOfCores(cudaDeviceProp devProp);


int main()
{
	int deviceCount;
	cudaDeviceProp deviceProp;

	cudaGetDeviceCount(&deviceCount);

	printf("Device count: %d\n\n", deviceCount);

	for (int i = 0; i < deviceCount; i++)
	{
		cudaGetDeviceProperties(&deviceProp, i);

		printf("Device name: %s\n", deviceProp.name);
		printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);

		printf("Clock rate: %d Hz\n", deviceProp.clockRate);
		printf("Multiprocessor count: %d\n", deviceProp.multiProcessorCount);
		printf("Number of cores: %d\n", getNumberOfCores(deviceProp));

		printf("Total constant memory: %d bytes\n", deviceProp.totalConstMem);
		printf("Total global memory: %zd bytes\n", deviceProp.totalGlobalMem);
		printf("Shared memory per block: %d bytes\n", deviceProp.sharedMemPerBlock);
		printf("Registers per block: %d\n", deviceProp.regsPerBlock);
		printf("Warp size: %d\n", deviceProp.warpSize);
		printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);

		printf("Max threads dimensions: x = %d, y = %d, z = %d\n",
			deviceProp.maxThreadsDim[0],
			deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[2]);

		printf("Max grid size: x = %d, y = %d, z = %d\n",
			deviceProp.maxGridSize[0],
			deviceProp.maxGridSize[1],
			deviceProp.maxGridSize[2]);
	}

	system("pause");

	return 0;
}


int getNumberOfCores(cudaDeviceProp devProp)
{
	int cores = 0;
	int mp = devProp.multiProcessorCount;
	switch (devProp.major) {
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
		if (devProp.minor == 1) cores = mp * 128;
		else if (devProp.minor == 0) cores = mp * 64;
		else printf("Unknown device type\n");
		break;
	default:
		printf("Unknown device type\n");
		break;
	}
	return cores;
}
