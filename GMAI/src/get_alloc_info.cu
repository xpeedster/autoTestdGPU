#include <stdio.h>
#include <stdlib.h>

__global__ void emptyKernel()
{
}

#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      printf("CUDA error: %s - %s(%d)\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main(int argc, char* argv[])
{
	// Initialization
	/*----------------------------------------------------------------------------------------*/
	int device = atoi(argv[1]);
	cudaCheck(cudaSetDevice(device));
	cudaSetDeviceFlags(cudaDeviceMapHost);

	int runtime_version;
	int driver_version;
	cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, device);
	cudaRuntimeGetVersion(&runtime_version);
	cudaDriverGetVersion(&driver_version);

	emptyKernel<<<1,1>>>();
	cudaDeviceSynchronize();
	
	// Pool size
	/*----------------------------------------------------------------------------------------*/
	int pool_size = 1;
	char *h_data;
	break_01: __attribute__((unused));
	cudaMallocManaged((void**)&h_data, 1 );
	 cudaFree(h_data);
	
	// Maximum allocations and granularity
	/*----------------------------------------------------------------------------------------*/
	char **h_data_array = (char**) malloc(pool_size * sizeof(char*));
	cudaMallocManaged((void**)&h_data_array[0], 1 );
	break_02: __attribute__((unused));
	int granularity = 0, iteration = 0, flag = 0;
	while(!flag && iteration < pool_size)
	{
		iteration++; 
		cudaMallocManaged((void**)&h_data_array[iteration], 1 );
	}
	for(int i = 0; i <= iteration; i++)
	{
		 cudaFree(h_data_array[i]);
	}
	free(h_data_array);
	
	// Size classes
	/*----------------------------------------------------------------------------------------*/
	char *h_data_inf, *h_data_sup;
	int inf_size = granularity, sup_size = granularity, finished = 1, class_finished = 0;
	break_03: __attribute__((unused));
	cudaMallocManaged((void**)&h_data_inf, inf_size );
	while(!finished)
	{
		sup_size = sup_size + granularity;
		cudaMallocManaged((void**)&h_data_sup, sup_size );
		 cudaFree(h_data_sup);
		if(class_finished)
		{
			class_finished = 0;
			 cudaFree(h_data_inf);
			inf_size = sup_size;
			cudaMallocManaged((void**)&h_data_inf, inf_size );
		}
	}
	 cudaFree(h_data_inf);
	
	// Larger allocations
	/*----------------------------------------------------------------------------------------*/
	break_04: __attribute__((unused));
	cudaMallocManaged((void**)&h_data, pool_size + 1 );
	 cudaFree(h_data);

	// Allocator policy
	/*----------------------------------------------------------------------------------------*/
	char *chunk_1, *chunk_2, *chunk_3, *chunk_4, *chunk_5, *chunk_6, *chunk_7, *chunk_8, *chunk_9, *chunk_10;
	cudaMallocManaged((void**)&chunk_1, granularity * 2 );
	cudaMallocManaged((void**)&chunk_2, granularity );
	cudaMallocManaged((void**)&chunk_3, granularity * 2 );
	cudaMallocManaged((void**)&chunk_4, granularity );
	cudaMallocManaged((void**)&chunk_5, granularity );
	cudaMallocManaged((void**)&chunk_6, granularity );
	 cudaFree(chunk_1);
	 cudaFree(chunk_3);
	 cudaFree(chunk_5);
	cudaMallocManaged((void**)&chunk_7, granularity );
	cudaMallocManaged((void**)&chunk_8, granularity );
	break_05: __attribute__((unused));
	 cudaFree(chunk_2);
	 cudaFree(chunk_4);
	 cudaFree(chunk_6);
	 cudaFree(chunk_7);
	 cudaFree(chunk_8);

	// Coalescing support
	/*----------------------------------------------------------------------------------------*/
	cudaMallocManaged((void**)&chunk_1, granularity );
	cudaMallocManaged((void**)&chunk_2, granularity );
	cudaMallocManaged((void**)&chunk_3, granularity );
	 cudaFree(chunk_1);
	 cudaFree(chunk_2);
	cudaMallocManaged((void**)&chunk_4, granularity * 2 );
	break_06: __attribute__((unused));
	 cudaFree(chunk_3);
	 cudaFree(chunk_4);

	// Splitting support
	/*----------------------------------------------------------------------------------------*/
	cudaMallocManaged((void**)&chunk_1, granularity * 2 );
	cudaMallocManaged((void**)&chunk_2, granularity );
	 cudaFree(chunk_1);
	cudaMallocManaged((void**)&chunk_3, granularity );
	break_07: __attribute__((unused));
	 cudaFree(chunk_2);
	 cudaFree(chunk_3);

	// Expansion policy
	/*----------------------------------------------------------------------------------------*/
	int max_allocations = pool_size / granularity;
	h_data_array = (char**) malloc(max_allocations * sizeof(char*));
	cudaMallocManaged((void**)&h_data_array[0], granularity );
	break_08: __attribute__((unused));
	int index;
	for(index = 1; index < max_allocations; index++)
	{
		cudaMallocManaged((void**)&h_data_array[index], granularity );
	}
	for(index = 0; index < max_allocations; index++)
	{
		 cudaFree(h_data_array[index]);
	}
	free(h_data_array);


	// Pool usage
	/*----------------------------------------------------------------------------------------*/
	int quarter = pool_size / 4;
	cudaMallocManaged((void**)&chunk_1, quarter );
	cudaMallocManaged((void**)&chunk_2, quarter );
	cudaMallocManaged((void**)&chunk_3, quarter );
	cudaMallocManaged((void**)&chunk_4, quarter );
	cudaMallocManaged((void**)&chunk_5, quarter );
	cudaMallocManaged((void**)&chunk_6, quarter );
	cudaMallocManaged((void**)&chunk_7, quarter );
	cudaMallocManaged((void**)&chunk_8, quarter );
	cudaMallocManaged((void**)&chunk_9, quarter );
	 cudaFree(chunk_1);
	 cudaFree(chunk_2);
	 cudaFree(chunk_5);
	cudaMallocManaged((void**)&chunk_10, quarter );
	break_09: __attribute__((unused));
	 cudaFree(chunk_10);

	// Shrinking support
	/*----------------------------------------------------------------------------------------*/
	flag = 0;
	break_10: __attribute__((unused));
	 cudaFree(chunk_6);
	 cudaFree(chunk_7);
	 cudaFree(chunk_8);
	flag = 1;
	 cudaFree(chunk_9);
	flag = 2;
	 cudaFree(chunk_3);
	 cudaFree(chunk_4);

	// Finalization
	/*----------------------------------------------------------------------------------------*/
	cudaDeviceReset();
	return 0;
}