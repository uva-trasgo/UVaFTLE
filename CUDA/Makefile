# Compilers
CC=${CUDA_HOME}/bin/nvcc

# Flags
FLAGS=-O3 -lm
CUDA_FLAGS=-lcusolver
FLAG_ZEN3=-march=znver3
FLAG_OMP=-fopenmp

# Directories
DIR=.
DIR_src=${DIR}/src
DIR_bin=${DIR}/bin

# Complementary files
SRC=${DIR_src}/preprocess.cu ${DIR_src}/arithmetic.cu

# Make lists
all: compute_ftle

# -------------------------- #
# ---------- SCALE --------- #
# -------------------------- #

compute_ftle:
	${CC} ${DIR_src}/ftle.cu ${SRC} -arch=sm_86 -Xcompiler ${FLAG_OMP} -I${CUDA_HOME}/include -L${CUDA_HOME}/lib -I ./include -o ${DIR_bin}/ftle_cuda_pinned -DPINNED ${FLAGS} ${CUDA_FLAGS}
	${CC} ${DIR_src}/ftle.cu ${SRC} -arch=sm_86 -Xcompiler ${FLAG_OMP} -I${CUDA_HOME}/include -L${CUDA_HOME}/lib -I ./include -o ${DIR_bin}/ftle_cuda ${FLAGS} ${CUDA_FLAGS}

clean:
	cd ${DIR_bin} && rm ${OBJS} && cd ..
