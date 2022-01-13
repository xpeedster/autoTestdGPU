CC = gcc
CXX = g++
NVCC = nvcc
ARCH ?= sm_50
NVPROF = "$(shell which nvprof)"

maj_flt-streams:
	-@$(NVCC) -arch $(ARCH) -g -G maj_flt-streams.cu --Wno-deprecated-gpu-targets -o maj_flt-streams -lcudart

.PHONY: test profile time run trace clean

test: maj_flt-streams
	-@rm -f pf_out1.txt
	-@sudo ./maj_flt-streams > pf_out1.txt
	-@sed -i"" 's/MAJFL//g' pf_out1.txt
	-@sed -ri '/^\s*$$/d' pf_out1.txt

profile: maj_flt-streams
	-@sudo /usr/local/cuda-10.2/bin/nvprof --print-gpu-trace ./maj_flt-streams

time: maj_flt-streams
	-@sudo /usr/bin/time -v ./maj_flt-streams

run: maj_flt-streams
	-@sudo $(NVPROF) --export-profile prof_streams ./maj_flt-streams > page-faults-streams.txt

trace: maj_flt-streams
	-@strace -f -e trace=mmap,munmap,open,mremap,ioctl,write -o maj_flt-streams.txt ./maj_flt-streams

clean:
	-@rm -f maj_flt-streams maj_flt-streams.txt prof_streams page-faults-streams.txt