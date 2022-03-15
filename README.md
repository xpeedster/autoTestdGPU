# autoTest dGPU

## How to launch

Para realizar las pruebas hay que lanzar el archivo "runtest.sh" de la siguiente forma

. ./runtest.sh sm_XX

Donde XX es el compute-capability de la GPU que se vaya a utilizar:

sm_72 para Jetson AGX Xavier <br />
sm_70 para TITAN V <br />
sm_61 para 1080Ti <br />
sm_50 para Quadro K1200 <br />

## Tests

### Hints
Tries different combinations of the hints. Locating data in CPU/GPU and advising the access of data from CPU/GPU. Also combining with memlocking. <br />
The Makefile requires the following arguments: 
ARCH <br />
SIZE (bytes) <br />
LOC (PreferredLocation hint: CPU = 1, GPU = 0, NONE = -1) <br />
ACC (AccessedBy hint: CPU = 1, GPU = 0, NONE = -1) <br />
MEM (memlockall: YES = 1, NO = 0) <br />

### Iterations and Sizes
Analysis of Unified Memory's behavior when using different variable sizes and allocating the data during more or less iterations. <br />
The Makefile requires the following arguments: 
ARCH <br />
SIZE (bytes) <br />
ITERS (Number of iterations) <br />
MEM (memlockall: YES = 1, NO = 0) <br />

### Access half
Same test as "Hints" but only accessing half of the allocated data. <br />
The Makefile requires the following arguments: 
ARCH <br />
SIZE (bytes) <br />
LOC (PreferredLocation hint: CPU = 1, GPU = 0, NONE = -1) <br />
ACC (AccessedBy hint: CPU = 1, GPU = 0, NONE = -1) <br />
MEM (memlockall: YES = 1, NO = 0) <br />

### Read and Write
Same test as "Hints" but we use two variables, one for read operations and the second one for write operations: A[i] = B[i] + 1; <br />
The Makefile requires the following arguments: 
ARCH <br />
SIZE (bytes) <br />
LOC (PreferredLocation hint: CPU = 1, GPU = 0, NONE = -1) <br />
ACC (AccessedBy hint: CPU = 1, GPU = 0, NONE = -1) <br />
MEM (memlockall: YES = 1, NO = 0) <br />

### Timing together
This test measures (using CUDA events) the time it takes for the whole access to conclude. <br />
The Makefile requires the following arguments: 
ARCH <br />
SIZE (bytes) <br />
LOC (PreferredLocation hint: CPU = 1, GPU = 0, NONE = -1) <br />
ACC (AccessedBy hint: CPU = 1, GPU = 0, NONE = -1) <br />
MEM (memlockall: YES = 1, NO = 0) <br />

### Timing split
This test measures (using CUDA events) the time it takes for the CPU access to conclude and the GPU access to conclude, separately. <br />
The Makefile requires the following arguments: 
ARCH <br />
SIZE (bytes) <br />
LOC (PreferredLocation hint: CPU = 1, GPU = 0, NONE = -1) <br />
ACC (AccessedBy hint: CPU = 1, GPU = 0, NONE = -1) <br />
MEM (memlockall: YES = 1, NO = 0) <br />

### IOCTL and MMAP
This test extracts the IOCTL and MMAP system calls used by Unified Memory. <br />
ARCH <br />
