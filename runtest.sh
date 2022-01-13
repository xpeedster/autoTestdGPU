export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Clean results folder
sudo rm -r ./results/*

# Test 1: IOCTL and MMAP
echo "Running test 1/4: IOCTL and MMAP"
cd ioctl-mmap
make clean && make trace ARCH=$1
cd ..

echo "Completed test 1/4: IOCTL and MMAP"
echo " "

# Test streams
echo "Streams 1/4"
cd args_test_streams
make clean && make run ARCH=$1 FILE=1 SIZE=65536 ITERS=25
mv prof_streams* ../results 
mv *.txt ../results
cd ..

echo "Streams 2/4"
cd args_test_streams
make clean && make run ARCH=$1 FILE=2 SIZE=65536 ITERS=100
mv prof_streams* ../results 
mv *.txt ../results
cd ..

echo "Streams 3/4"
cd args_test_streams
make clean && make run ARCH=$1 FILE=3 SIZE=131072 ITERS=25
mv prof_streams* ../results 
mv *.txt ../results
cd ..

echo "Streams 4/4"
cd args_test_streams
make clean && make run ARCH=$1 FILE=4 SIZE=131072 ITERS=100
mv prof_streams* ../results 
mv *.txt ../results
cd ..
echo " "

# Test hints resident in CPU

echo "Hints-CPU 1/4"
cd args_test_hintsCPU
make clean && make run2 ARCH=$1 FILE=1 SIZE=65536 ACBY=1
mv prof_hintsCPU* ../results 
mv *.txt ../results
cd ..

echo "Hints-CPU 2/4"
cd args_test_hintsCPU
make clean && make run2 ARCH=$1 FILE=2 SIZE=65536 ACBY=0
mv prof_hintsCPU* ../results 
mv *.txt ../results
cd ..

echo "Hints-CPU 3/4"
cd args_test_hintsCPU
make clean && make run2 ARCH=$1 FILE=3 SIZE=131072 ACBY=1
mv prof_hintsCPU* ../results 
mv *.txt ../results
cd ..

echo "Hints-CPU 4/4"
cd args_test_hintsCPU
make clean && make run2 ARCH=$1 FILE=4 SIZE=131072 ACBY=0
mv prof_hintsCPU* ../results 
mv *.txt ../results
cd ..
echo " "

# Test hints resident in GPU

echo "Hints-GPU 1/4"
cd args_test_hintsGPU
make clean && make run2 ARCH=$1 FILE=1 SIZE=65536 ACBY=1
mv prof_hintsGPU* ../results 
mv *.txt ../results
cd ..

echo "Hints-GPU 2/4"
cd args_test_hintsGPU
make clean && make run2 ARCH=$1 FILE=2 SIZE=65536 ACBY=0
mv prof_hintsGPU* ../results 
mv *.txt ../results
cd ..

echo "Hints-GPU 3/4"
cd args_test_hintsGPU
make clean && make run2 ARCH=$1 FILE=3 SIZE=131072 ACBY=1
mv prof_hintsGPU* ../results 
mv *.txt ../results
cd ..

echo "Hints-GPU 4/4"
cd args_test_hintsGPU
make clean && make run2 ARCH=$1 FILE=4 SIZE=131072 ACBY=0
mv prof_hintsGPU* ../results 
mv *.txt ../results
cd ..
echo " "

# Use GMAI
echo "Using GMAI on GPU"
cd GMAI/src
make clean_info && make show_alloc_info arch=$1
cp info.cfg ../../results
cd ../..
echo " "


echo " "
echo "All tests completed. Please compress the folder and send back the results."