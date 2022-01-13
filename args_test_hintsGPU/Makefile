CC = gcc
CXX = g++
NVCC = nvcc
ARCH ?= sm_50
NVPROF = "$(shell which nvprof)"
FILE ?= 1
SIZE ?= 65536
ACBY ?= 1

hints-GPU:
	-@$(NVCC) -arch $(ARCH) -g -G hints-GPU.cu --Wno-deprecated-gpu-targets -o hints-GPU -lcudart

.PHONY: run trace clean

run: hints-GPU
	-@./hints-GPU > hints-resident-GPU.txt

run2: hints-GPU
	-@sudo $(NVPROF) --export-profile prof_hintsGPU$(FILE) ./hints-GPU $(SIZE) $(ACBY) > hints-resident-GPU$(FILE).txt

trace: hints-GPU
	-@strace -f -e trace=mmap,munmap,open,mremap,ioctl,write -o hints-GPU.txt ./hints-GPU > hintsAttributes.txt

clean:
	-@rm -f hints-GPU *.txt prof_hintsGPU*