export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Test 1: IOCTL and MMAP
cd ioctl-mmap
make clean
cd ..

echo "Completed test 1/4: IOCTL and MMAP"

# Test 2: Hints with data resident on CPU and GPU
cd hints-concurrent-resident-CPU
make clean
cd ..

cd hints-concurrent-resident-GPU
make clean
cd ..

echo "Completed test 2/4: Hints with data resident on CPU and GPU. This might take long..."

# Test 3: Page faults (patterns and ranges)
cd page-faults-patterns
make clean
cd ..

cd page-faults-ranges
make clean
cd ..

echo "Completed test 3/4: Page faults (patterns and ranges)"

# Test 4: Default stream and stream1
cd page-faults-streams
make clean
cd ..

echo "Completed test 4/4: Default stream and stream1"

echo " "
echo "All tests completed. Please compress the folder and send back the results."