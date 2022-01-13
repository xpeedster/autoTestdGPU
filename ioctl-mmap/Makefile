CC = gcc
CXX = g++
NVCC = nvcc
ARCH ?= sm_50

ioctl-mmap:
	-@$(NVCC) -arch $(ARCH) -g -G ioctl-mmap.cu --Wno-deprecated-gpu-targets -o ioctl-mmap -lcudart

.PHONY: run trace clean

run: ioctl-mmap
	-@./ioctl-mmap

trace: ioctl-mmap
	-@strace -f -e trace=mmap,munmap,open,mremap,ioctl,write -o ioctl-mmap.txt ./ioctl-mmap > ioctl-mmap-cmd-out.txt

clean:
	-@rm -f ioctl-mmap ioctl-mmap.txt ioctl-mmap-cmd-out.txt