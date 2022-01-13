#include <stdio.h>
#include <iostream>

#define CHECK(call)															\
{																			\
	const cudaError_t error = call;											\
	if(error != cudaSuccess)												\
	{																		\
		printf("Error: %s:%d, ", __FILE__, __LINE__);						\
		printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));	\
		exit(1);															\
	}																		\
}		

__global__ void emptyKernel()
{
}

int main()
{
	//cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceScheduleSpin);
	emptyKernel<<<1,1>>>();
	cudaDeviceSynchronize();
	char *A = 0, *B = 0;
	int N;
	
	N = 1;
	std::cout << "First allocation of " << N << " B" << std::endl;
    cudaMallocManaged(&A, N);
    std::cout << "Second allocation of " << N << " B" << std::endl;
	cudaMallocManaged(&B, N);
	std::cout << "First free" << std::endl;
    cudaFree(A);
    std::cout << "First free" << std::endl;
	cudaFree(B);
	
	N = 65536;
    std::cout << "First allocation of " << N << " B" << std::endl;
    cudaMallocManaged(&A, N);
    std::cout << "Second allocation of " << N << " B" << std::endl;
	cudaMallocManaged(&B, N);
	std::cout << "First free" << std::endl;
    cudaFree(A);
    std::cout << "First free" << std::endl;
	cudaFree(B);

	N = 1048576;
	std::cout << "First allocation of " << N << " B" << std::endl;
    cudaMallocManaged(&A, N);
    std::cout << "Second allocation of " << N << " B" << std::endl;
	cudaMallocManaged(&B, N);
	std::cout << "First free" << std::endl;
    cudaFree(A);
    std::cout << "First free" << std::endl;
	cudaFree(B);

	N = 2097152;
	std::cout << "First allocation of " << N << " B" << std::endl;
    cudaMallocManaged(&A, N);
    std::cout << "Second allocation of " << N << " B" << std::endl;
	cudaMallocManaged(&B, N);
	std::cout << "First free" << std::endl;
    cudaFree(A);
    std::cout << "First free" << std::endl;
	cudaFree(B);

	N = 2097156;
	std::cout << "First allocation of " << N << " B" << std::endl;
    cudaMallocManaged(&A, N);
    std::cout << "Second allocation of " << N << " B" << std::endl;
	cudaMallocManaged(&B, N);
	std::cout << "First free" << std::endl;
    cudaFree(A);
    std::cout << "First free" << std::endl;
	cudaFree(B);


	std::cout << "----------------------------------------\n" ;

	cudaDeviceReset();

}
