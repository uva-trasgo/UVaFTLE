#!/bin/bash

cd ..
aocl initialize acl0 pac_s10
icpx    -fsycl -fintelfpga -c -o arithmetic.o src/arithmetic.cpp -I./include
icpx    -fsycl -fintelfpga -c -o preprocess.o src/preprocess.cpp -I./include
icpx    -fsycl -fintelfpga -c -o ftle.o src/ftle.cpp -I./include
time icpx    -fsycl -fintelfpga -Xshardware -Xsboard=intel_s10sx_pac:pac_s10 arithmetic.o  preprocess.o ftle.o -o ftle_fpga -reuse-exe=ftle_fpga

#icpx    -fsycl -fintelfpga -c -o ftle.o src/ftle.cpp -I./include
