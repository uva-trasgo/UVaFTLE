# Flags

# Make lists
all: compute_ftle

# -------------------------- #
# ---------- GCC ----------- #
# -------------------------- #

compute_ftle:
	hipcc hip_code.cpp -fopenmp -lm -std=c++11 -o flte_hip
	hipcc hip_pinned_code.cpp -fopenmp -lm -std=c++11 -o flte_hip_pinned
	hipcc hip_managed_code.cpp -fopenmp -lm -std=c++11 -o flte_hip_managed
	nvcc cuda_code.cu -arch=sm_70 -Xcompiler -fopenmp -o ftle_cuda
	nvcc cuda_pinned_code.cu -arch=sm_70 -Xcompiler -fopenmp -o ftle_cuda_pinned
	nvcc cuda_managed_code.cu -arch=sm_70 -Xcompiler -fopenmp -o ftle_cuda_managed
	acpp -O3 --acpp-targets='generic' sycl_buffers1.cpp -o ftle_buffers1	
	acpp -O3 --acpp-targets='generic' sycl_buffers2.cpp -o ftle_buffers2
	acpp -O3 --acpp-targets='generic' sycl_buffers4.cpp -o ftle_buffers4
	acpp -O3 --acpp-targets='generic' sycl_buffers8.cpp -o ftle_buffers8
	acpp -O3 --acpp-targets='generic' sycl_usm_shared.cpp -o ftle_usm_shared
	acpp -O3 --acpp-targets='generic' sycl_usm_device.cpp -o ftle_usm_device
clean:
	rm ${OBJS}
