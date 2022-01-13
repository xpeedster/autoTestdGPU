CXX = g++
NVCC = nvcc
arch ?= sm_50
device ?= 0
program_name ?=
program_arguments ?=
verbosity ?= 0

get_alloc_info:
	-@$(NVCC) -arch=$(arch) -g get_alloc_info.cu -o get_alloc_info

info.cfg: get_alloc_info
	-@gdb -batch-silent -iex "set auto-load safe-path /" --args get_alloc_info $(device) 2>/dev/null

print_alloc_info: info.cfg
	-@$(CXX) AllocatorInfo.cpp print_alloc_info.cpp -o print_alloc_info

preload_library.so:
	-@$(CXX) -I/usr/local/cuda/include -shared -fPIC MemoryPool.cpp SizeClass.cpp AllocatorInfo.cpp AllocatorStatus.cpp preload_library.cpp -o preload_library.so -ldl

.PHONY: show_alloc_info memory_analysis clean clean_info

show_alloc_info: print_alloc_info
	-@./print_alloc_info

memory_analysis: info.cfg preload_library.so
	-@$(NVCC) -arch=$(arch) -cudart shared $(program_name).cu -o $(program_name)
	-@VERBOSITY=$(verbosity) LD_PRELOAD="preload_library.so" ./$(program_name) $(program_arguments) 2>$(program_name).txt
	-@cat $(program_name).txt

clean:
	-@rm -f preload_library.so

clean_info:
	-@rm -f get_alloc_info print_alloc_info info.cfg
