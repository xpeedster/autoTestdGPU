CC = gcc
CXX = g++
NVCC = nvcc
ARCH ?= sm_50
NVPROF = "$(shell which nvprof)"
FILE ?= 1
SIZE ?= 65536
ACBY ?= 1

hints-CPU:
	-@$(NVCC) -arch $(ARCH) -g -G hints-CPU.cu --Wno-deprecated-gpu-targets -o hints-CPU -lcudart

.PHONY: run trace clean

run: hints-CPU
	-@./hints-CPU > hints-resident-CPU.txt

run2: hints-CPU
	-@sudo $(NVPROF) --export-profile prof_hintsCPU$(FILE) ./hints-CPU $(SIZE) $(ACBY) > hints-resident-CPU$(FILE).txt

trace: hints-CPU
	-@strace -f -e trace=mmap,munmap,open,mremap,ioctl,write -o hints-CPU.txt ./hints-CPU > hintsAttributes.txt

clean:
	-@rm -f hints-CPU *.txt prof_hintsCPU*