# Flags

# Make lists
all: compute_ftle

# -------------------------- #
# ---------- GCC ----------- #
# -------------------------- #

compute_ftle:
	#nvcc cuda_code.cu  -o ftle_cuda -lcusolver -Xcompiler -fopenmp -allow-unsupported-compiler -arch=sm_70 -Xcompiler -O3 -lm -I/opt/cuda-11.3/include/ -L/opt/cuda-11.3/lib
	syclcc    -O3  --hipsycl-explicit-multipass --hipsycl-targets='hip:gfx900;cuda:sm_70' --hipsycl-cpu-cxx=/home/trasgo/local/llvm14/bin/clang++ --rocm-path=/opt/rocm -I/opt/rocm/hip/include/ -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/oclc_abi_version_400.bc sycl_code8.cpp -o ftle_sycl8
	syclcc    -O3  --hipsycl-explicit-multipass --hipsycl-targets='hip:gfx900;cuda:sm_70' --hipsycl-cpu-cxx=/home/trasgo/local/llvm14/bin/clang++ --rocm-path=/opt/rocm -I/opt/rocm/hip/include/ -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/oclc_abi_version_400.bc sycl_code1.cpp -o ftle_sycl1	
	syclcc    -O3  --hipsycl-explicit-multipass --hipsycl-targets='hip:gfx900;cuda:sm_70' --hipsycl-cpu-cxx=/home/trasgo/local/llvm14/bin/clang++ --rocm-path=/opt/rocm -I/opt/rocm/hip/include/ -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/oclc_abi_version_400.bc sycl_code4.cpp -o ftle_sycl4
	syclcc    -O3  --hipsycl-explicit-multipass --hipsycl-targets='hip:gfx900;cuda:sm_70' --hipsycl-cpu-cxx=/home/trasgo/local/llvm14/bin/clang++ --rocm-path=/opt/rocm -I/opt/rocm/hip/include/ -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/oclc_abi_version_400.bc sycl_code2.cpp -o ftle_sycl2
	hipcc hip_code.cpp -fopenmp  -I/usr/local/cppda/include/ -L/usr/local/cppda/bin/ -I ./include -L/opt/rocm-5.4.0/lib64  -L/opt/rocm-5.4.0/ -O3 -lm -std=c++11 -o flte_hip
clean:
	rm ${OBJS}
