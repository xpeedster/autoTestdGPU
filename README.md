# autoTest dGPU

## How to launch

Para realizar las pruebas hay que lanzar el archivo "runtest.sh" de la siguiente forma

. ./runtest.sh sm_XX

Donde XX es el compute-capability de la GPU que se vaya a utilizar:

sm_72 para Jetson AGX Xavier
sm_70 para TITAN V
sm_61 para 1080Ti
sm_50 para Quadro K1200

El test se da por concluído al recebir el siguiente output:

    Completed test 1/4: IOCTL and MMAP
    Completed test 2/4: Hints with data resident on CPU and GPU. This might take long...
    Completed test 3/4: Page faults (patterns and ranges)
    ==19800== NVPROF is profiling process 19800, command: ./maj_flt-streams
    ==19800== Generated result file: /home/xarauzo/UnifiedMem/automaticTestdGPU/page-faults-streams/prof_streams
    Completed test 4/4: Default stream and stream1
     
    All tests completed. Please compress the folder and send back the results.