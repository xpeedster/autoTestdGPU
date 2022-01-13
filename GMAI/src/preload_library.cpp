#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <cuda_runtime.h>
#include "AllocatorInfo.h"
#include "AllocatorStatus.h"

using namespace std;

typedef cudaError_t (*real_cudamalloc_t)(void **devPtr, size_t size);
typedef cudaError_t (*real_cudahostalloc_t)(void **pHost, size_t size, unsigned int flags);
typedef cudaError_t (*real_cudafree_t)(void *devPtr);
typedef cudaError_t (*real_cudafreehost_t)(void *ptr);

static AllocatorStatus *deviceAllocatorStatus;
static AllocatorStatus *hostAllocatorStatus;
static int verbose;
static int counter = 0;

void __attribute__ ((constructor)) lib_init(void)
{
    AllocatorInfo info("info.cfg");
    verbose = atoi(getenv("VERBOSITY"));
    deviceAllocatorStatus = new AllocatorStatus(info);
    hostAllocatorStatus = new AllocatorStatus(info);
}

void __attribute__ ((destructor)) lib_fini(void)
{
    if(deviceAllocatorStatus->getMaximumUserMemory() > 0)
    {
        fprintf(stderr, "Device memory:\n");
        deviceAllocatorStatus->printMaximum();
    }

    if(hostAllocatorStatus->getMaximumUserMemory() > 0)
    {
        fprintf(stderr, "Host memory:\n");
        hostAllocatorStatus->printMaximum();
    }
    
    delete(deviceAllocatorStatus);
    delete(hostAllocatorStatus);
}

cudaError_t cudaMalloc(void **devPtr, size_t size)
{
    cudaError_t error = ((real_cudamalloc_t)dlsym(RTLD_NEXT, "cudaMalloc"))(devPtr, size);
    deviceAllocatorStatus->addAllocation(*devPtr, size);
    if(verbose)
    {
        fprintf(stderr, "%d. cudaMalloc(%lu):\n", ++counter, size);
        deviceAllocatorStatus->printStatus();
    }
    return error;
}

cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags)
{
    cudaError_t error = ((real_cudahostalloc_t)dlsym(RTLD_NEXT, "cudaHostAlloc"))(pHost, size, flags);
    hostAllocatorStatus->addAllocation(*pHost, size);
    if(verbose)
    {
        fprintf(stderr, "%d. cudaHostAlloc(%lu):\n", ++counter, size);
        hostAllocatorStatus->printStatus();
    }
    return error;
}

cudaError_t cudaFree(void *devPtr)
{
    cudaError_t error = ((real_cudafree_t)dlsym(RTLD_NEXT, "cudaFree"))(devPtr);
    deviceAllocatorStatus->removeAllocation(devPtr);
    if(verbose)
    {
        fprintf(stderr, "%d. cudaFree():\n", ++counter);
        deviceAllocatorStatus->printStatus();
    }
    return error;
} 

cudaError_t cudaFreeHost(void *ptr)
{
    cudaError_t error = ((real_cudafreehost_t)dlsym(RTLD_NEXT, "cudaFreeHost"))(ptr);
    hostAllocatorStatus->removeAllocation(ptr);
    if(verbose)
    {
        fprintf(stderr, "%d. cudaFreeHost():\n", ++counter);
        hostAllocatorStatus->printStatus();
    }
    return error;
}