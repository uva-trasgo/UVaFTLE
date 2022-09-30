# Compilers
CC=nvcc

# Flags
FLAGS=-O3 -lm 
FLAG_ZEN3=-march=znver3
FLAG_OMP=-Xcompiler -fopenmp

# Directories
DIR=.
DIR_src=${DIR}/src
DIR_bin=${DIR}/bin

# Complementary files
SRC=${DIR_src}/preprocess.cu ${DIR_src}/arithmetic.cu

# Make lists
all: compute_ftle

# -------------------------- #
# ---------- GCC ----------- #
# -------------------------- #

compute_ftle:
	${CC} ${DIR_src}/ftle_static.cu ${SRC} ${FLAG_OMP} -I ./include -o ${DIR_bin}/ftle_static ${FLAGS}
	${CC} ${DIR_src}/ftle_dynamic.cu ${SRC} ${FLAG_OMP} -I ./include -o ${DIR_bin}/ftle_dynamic ${FLAGS}
	${CC} ${DIR_src}/ftle_guided.cu ${SRC} ${FLAG_OMP} -I ./include -o ${DIR_bin}/ftle_guided ${FLAGS}

clean:
	cd ${DIR_bin} && rm ${OBJS} && cd ..
