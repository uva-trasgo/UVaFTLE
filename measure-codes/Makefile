# Flags

# Make lists
all: compute_ftle

# -------------------------- #
# ---------- GCC ----------- #
# -------------------------- #

compute_ftle:
	acpp -O3 --acpp-targets='generic' sycl_buffers1.cpp -o ftle_buffers1	
	acpp -O3 --acpp-targets='generic' sycl_buffers2.cpp -o ftle_buffers2
	acpp -O3 --acpp-targets='generic' sycl_buffers4.cpp -o ftle_buffers4
	acpp -O3 --acpp-targets='generic' sycl_buffers8.cpp -o ftle_buffers8
	acpp -O3 --acpp-targets='generic' sycl_usm.cpp -o ftle_usm
	hipcc hip_code.cpp -fopenmp -lm -std=c++11 -o flte_hip
	nvcc cuda_code.cu -arch=sm_70 -Xcompiler -fopenmp -o ftle_cuda
clean:
	rm ${OBJS}
