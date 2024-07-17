# Compilerscppk#CC=gcc
CC=hipcc

# Flags
FLAGS=-O3 -lm -std=c++11
#cppDA_FLAGS=-lcppsolver
FLAG_ZEN3=-march=znver3 
FLAG_OMP=-fopenmp
#-arch=sm_35
# Directories
DIR=.
DIR_src=${DIR}/src
DIR_bin=${DIR}/bin
#HIPCC_VERBOSE=7
# Complementary files
SRC=${DIR_src}/preprocess.cpp ${DIR_src}/arithmetic.cpp

# Make lists
all: compute_ftle

# -------------------------- #
# ---------- GCC ----------- #
# -------------------------- #

compute_ftle:
	HIPCC_VERBOSE=7   ${CC}  ${DIR_src}/ftle.cpp ${SRC} -fopenmp  -D__HIP_PLATFORM_AMD__  --offload-arch=gfx900 -I ./include -L/opt/rocm/lib64  -L/opt/rocm/lib -DPINNED -o ${DIR_bin}/ftle_rocm_pin ${FLAGS} 
	HIPCC_VERBOSE=7   ${CC}  ${DIR_src}/ftle.cpp ${SRC} -fopenmp  -D__HIP_PLATFORM_AMD__  --offload-arch=gfx900 -I ./include -L/opt/rocm/lib64  -L/opt/rocm/lib -o ${DIR_bin}/ftle_rocm ${FLAGS} 
	
clean:
	cd ${DIR_bin} && rm ${OBJS} && cd ..
