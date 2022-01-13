#include <stdio.h>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <sstream>

#define CHECK(call)															\
{																			\
	const cudaError_t error = call;											\
	if(error != cudaSuccess)												\
	{																		\
		printf("Error: %s:%d, ", __FILE__, __LINE__);						\
		printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));	\
	}																		\
}

std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}


__global__ void emptyKernel()
{
}

__global__ void sumKernel_all(float *A, int elm)
{
	for(int i = 0; i < elm; i++){
		A[i] = A[i] + 1;
	}
}

void test_all(int size, float *A)
{
	int elm = size/sizeof(float);
	std::cout << "Access All the data using " << size << " bytes" << std::endl;
	cudaMallocManaged(&A, size);
	std::system("ps -C maj_flt-streams -o maj_flt,min_flt");
	sumKernel_all<<<1,1>>>(A, elm);
	cudaDeviceSynchronize();

	for(int i = 0; i < elm; i++){
		A[i] = A[i] + 1;
	}
	std::system("ps -C maj_flt-streams -o maj_flt,min_flt");
	cudaFree(A);

}

int test_sumKernel(int size, float *A, cudaStream_t stream){
	int elm = size/sizeof(float);
	//std::cout << "Size = " << size << " B. All data accessed. Page faults = " ;
	cudaMallocManaged(&A, size);
	std::string bef_string = exec("ps -C maj_flt-streams -o maj_flt");

	for(int t1 = 0; t1 < 100; t1++)
	{
		sumKernel_all<<<1,1,0,stream>>>(A, elm);
		cudaDeviceSynchronize();
	
		for(int i = 0; i < elm; i++){
			A[i] = A[i] + 1;
		}
	}

	std::string aft_string = exec("ps -C maj_flt-streams -o maj_flt");
	cudaFree(A);

	// Process strings
	std::stringstream ss_bef;
	std::stringstream ss_aft;
	bef_string.erase(0,9);
	aft_string.erase(0,9);
	ss_bef << bef_string;
	ss_aft << aft_string;
	int before, after, maj_flt;
	ss_bef >> before;
	ss_aft >> after;
	maj_flt = after - before;
	return maj_flt;
}

int test_empty(int size, float *A, cudaStream_t stream){
	int elm = size/sizeof(float);
	//std::cout << "Size = " << size << " B. All data accessed. Page faults = " ;
	cudaMallocManaged(&A, size);
	std::string bef_string = exec("ps -C maj_flt-streams -o maj_flt");

	for(int t1 = 0; t1 < 100; t1++)
	{
		emptyKernel<<<1,1,0,stream>>>();
		cudaDeviceSynchronize();
	
		for(int i = 0; i < elm; i++){
			A[i] = A[i] + 1;
		}
	}

	std::string aft_string = exec("ps -C maj_flt-streams -o maj_flt");
	cudaFree(A);

	// Process strings
	std::stringstream ss_bef;
	std::stringstream ss_aft;
	bef_string.erase(0,9);
	aft_string.erase(0,9);
	ss_bef << bef_string;
	ss_aft << aft_string;
	int before, after, maj_flt;
	ss_bef >> before;
	ss_aft >> after;
	maj_flt = after - before;
	return maj_flt;
}

int test_noGPU(int size, float *A){
	int elm = size/sizeof(float);
	//std::cout << "Size = " << size << " B. All data accessed. Page faults = " ;
	cudaMallocManaged(&A, size);
	std::string bef_string = exec("ps -C maj_flt-streams -o maj_flt");

	for(int t1 = 0; t1 < 100; t1++)
	{	
		for(int i = 0; i < elm; i++){
			A[i] = A[i] + 1;
		}
	}

	std::string aft_string = exec("ps -C maj_flt-streams -o maj_flt");
	cudaFree(A);

	// Process strings
	std::stringstream ss_bef;
	std::stringstream ss_aft;
	bef_string.erase(0,9);
	aft_string.erase(0,9);
	ss_bef << bef_string;
	ss_aft << aft_string;
	int before, after, maj_flt;
	ss_bef >> before;
	ss_aft >> after;
	maj_flt = after - before;
	return maj_flt;
}


int main()
{
	cudaStream_t stream1;
	cudaStreamCreate(&stream1);
	
	//cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceScheduleSpin);
	emptyKernel<<<1,1>>>();
	cudaDeviceSynchronize();
	float *A = 0;
	int N, faults;
	
	N = 65536;
	std::cout << "Tests done using allocations of size = " << N << " bytes." << std::endl;

	// Test 0: no GPU. Only CPU. 100 iterations.
	faults = test_noGPU(N, A);
	std::cout << "Test0: noGPU. 100 iterations from CPU." << std::endl;
	std::cout << "		PageFaults = " << faults << std::endl;

	// Test 1: sumKernel_all using stream0 with 100 iterations.
	faults = test_sumKernel(N, A, 0);
	std::cout << "Test1: Stream0 and sumKernel_all." << std::endl;
	std::cout << "		PageFaults = " << faults << std::endl;

	// Test 2: sumKernel_all using stream1 with 100 iterations.
	faults = test_sumKernel(N, A, stream1);
	std::cout << "Test2: Stream1 and sumKernel_all." << std::endl;
	std::cout << "		PageFaults = " << faults << std::endl;

	// Test 3: emptyKernel using stream0 with 100 iterations.
	faults = test_empty(N, A, 0);
	std::cout << "Test3: Stream0 and emptyKernel." << std::endl;
	std::cout << "		PageFaults = " << faults << std::endl;

	// Test 4: emptyKernel using stream1 with 100 iterations.
	faults = test_empty(N, A, stream1);
	std::cout << "Test4: Stream1 and emptyKernel." << std::endl;
	std::cout << "		PageFaults = " << faults << std::endl;

	
	cudaDeviceReset();

}
