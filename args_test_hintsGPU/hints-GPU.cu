#include <stdio.h>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <sstream>

#include <sys/mman.h>

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

__global__ void sumKernel(float *A, int elm)
{
	for(int i = 0; i < elm; i++){
		A[i] = A[i] + 1;
	}
}


int main(int argc, char** argv)
{
	// Setup
	std::string pf_before;
	std::string pf_after;

	pf_before = exec("ps -C hints-GPU -o maj_flt");

	// CPU Affinity
	cpu_set_t mask;
	CPU_ZERO(&mask);
	CPU_SET(1, &mask);
	if(sched_setaffinity(0, sizeof(cpu_set_t), &mask) <0)
	{
		perror("sched_setaffinity_failed");
		exit(-1);
	}

	// Memory locking
	if(mlockall(MCL_CURRENT | MCL_FUTURE) < 0)
	{
		perror("mklockall failed");
		exit(-1);
	}

	// Test

	//cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceScheduleSpin);
	emptyKernel<<<1,1>>>();
	cudaDeviceSynchronize();
	float *A = 0;
	//int N = 65536;
	int N = atoi(argv[1]);
	//int it = atoi(argv[2]);
	int accBy = atoi(argv[2]);
	int elm = N/sizeof(float);
	printf("hints-GPU with size %d and AccessedBy\n", elm);

	int result;
    cudaDeviceGetAttribute (&result, cudaDevAttrConcurrentManagedAccess, 0);
	printf("cudaDevAttrConcurrentManagedAccess = %d\n", result);

    cudaMallocManaged(&A, N);

	CHECK(cudaMemAdvise(A, N, cudaMemAdviseSetPreferredLocation, 0));
	printf("cudaCPUDeviceId = %d\n", cudaCpuDeviceId);
	
	if(accBy == 1){
		printf("hints-GPU with size %d and AccessedBy\n", N);
		CHECK(cudaMemAdvise(A, N, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId));
	}else{
		printf("hints-GPU with size %d and NOT AccessedBy\n", N);
	}
	
	// Initialize data
	std::system("ps -C hints-GPU -o maj_flt");
	for(int i = 0; i < elm; i++){
		A[i] = 1;
	}
	std::system("ps -C hints-GPU -o maj_flt");
	
	// Allocated size is 64 KB, hence, 10 iterations should result in 10 maj_flts if data does NOT stay in GPU.

	// Run test empty kernel
	std::system("ps -C hints-GPU -o maj_flt");
	for(int i_emp = 0; i_emp < 10; i_emp++)
	{
		emptyKernel<<<1,1>>>();
		cudaDeviceSynchronize();
		
		for(int i = 0; i < elm; i++){
			A[i] = A[i] + 1;
		}
	}
	std::system("ps -C hints-GPU -o maj_flt");
	
	// Run test sum kernel
	std::system("ps -C hints-GPU -o maj_flt");
	for(int i_emp = 0; i_emp < 10; i_emp++)
	{
		sumKernel<<<1,1>>>(A, elm);
		cudaDeviceSynchronize();

		for(int i = 0; i < elm; i++){
			A[i] = A[i] + 1;
		}
	}
	std::system("ps -C hints-GPU -o maj_flt");
	
    cudaFree(A);

	pf_after = exec("ps -C hints-GPU -o maj_flt");

	// Process strings
	std::stringstream ss_pf_b;
	std::stringstream ss_pf_a;
	pf_before.erase(0,9);
	pf_after.erase(0,9);
	ss_pf_b << pf_before;
	ss_pf_a << pf_after;
	int before, after, maj_flt;
	ss_pf_b >> before;
	ss_pf_a >> after;
	maj_flt = after - before;
	
	std::cout << " Total Page faults during the execution = " << maj_flt << std::endl;

	cudaDeviceReset();

}
