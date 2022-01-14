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

__global__ void sumKernel_all(float *A, int elm)
{
	for(int i = 0; i < elm; i++){
		A[i] = A[i] + 1;
	}
}

int test_sumKernel(int size, float *A, cudaStream_t stream, int iters){
	int elm = size/sizeof(float);
	//std::cout << "Size = " << size << " B. All data accessed. Page faults = " ;
	cudaMallocManaged(&A, size);
	std::string bef_string = exec("ps -C maj_flt-streams -o maj_flt");

	for(int t1 = 0; t1 < iters; t1++)
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

int test_empty(int size, float *A, cudaStream_t stream, int iters){
	int elm = size/sizeof(float);
	//std::cout << "Size = " << size << " B. All data accessed. Page faults = " ;
	cudaMallocManaged(&A, size);
	std::string bef_string = exec("ps -C maj_flt-streams -o maj_flt");

	for(int t1 = 0; t1 < iters; t1++)
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

int test_noGPU(int size, float *A, int iters){
	int elm = size/sizeof(float);
	//std::cout << "Size = " << size << " B. All data accessed. Page faults = " ;
	cudaMallocManaged(&A, size);
	std::string bef_string = exec("ps -C maj_flt-streams -o maj_flt");

	for(int t1 = 0; t1 < iters; t1++)
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


int main(int argc, char** argv)
{
	std::string pf_before;
	std::string pf_after;

	pf_before = exec("ps -C maj_flt-streams -o maj_flt");

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


	cudaStream_t stream1;
	cudaStreamCreate(&stream1);
	
	//cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceScheduleSpin);
	emptyKernel<<<1,1>>>();
	cudaDeviceSynchronize();
	float *A = 0;
	int N, faults, iters;
	
	//N = 65536;
	N = atoi(argv[1]);
	iters = atoi(argv[2]);
	std::cout << "Tests done using allocations of size = " << N << " bytes." << std::endl;

	// Test 0: no GPU. Only CPU. 25 iterations.
	faults = test_noGPU(N, A, iters);
	std::cout << "Test0: noGPU. " << iters << " iterations from CPU." << std::endl;
	std::cout << "		PageFaults = " << faults << std::endl;

	// Test 1: sumKernel_all using stream0 with 25 iterations.
	faults = test_sumKernel(N, A, 0, iters);
	std::cout << "Test1: Stream0 and sumKernel_all. " << iters << " iterations." << std::endl;
	std::cout << "		PageFaults = " << faults << std::endl;

	// Test 2: sumKernel_all using stream1 with 25 iterations.
	faults = test_sumKernel(N, A, stream1, iters);
	std::cout << "Test2: Stream1 and sumKernel_all." << iters << " iterations." << std::endl;
	std::cout << "		PageFaults = " << faults << std::endl;

	// Test 3: emptyKernel using stream0 with 25 iterations.
	faults = test_empty(N, A, 0, iters);
	std::cout << "Test3: Stream0 and emptyKernel. " << iters << " iterations." << std::endl;
	std::cout << "		PageFaults = " << faults << std::endl;

	// Test 4: emptyKernel using stream1 with 25 iterations.
	faults = test_empty(N, A, stream1, iters);
	std::cout << "Test4: Stream1 and emptyKernel. " << iters << " iterations." << std::endl;
	std::cout << "		PageFaults = " << faults << std::endl;

	pf_after = exec("ps -C maj_flt-streams -o maj_flt");

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
