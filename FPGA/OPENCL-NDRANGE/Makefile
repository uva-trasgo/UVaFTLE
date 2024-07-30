# Compilers
CC=gcc

# Flags
FLAGS=-O3 -lm -Wall
OPENCL_FLAGS=-lOpenCL -I /opt/intel/oneapi/compiler/latest/linux/include/sycl -DOPENCL_DEBUG

# Directories
DIR=.
DIR_src=${DIR}/src
DIR_bin=${DIR}/bin

# Complementary files
SRC=${DIR_src}/preprocess.c ${DIR_src}/opencl_helper.c

# Make lists
all: compute_ftle
emulation: compute_ftle_emu
emu: compute_ftle_emu

# -------------------------- #
# ---------- GCC ----------- #
# -------------------------- #

compute_ftle:
	${CC} ${DIR_src}/ftle.c ${SRC} -I ./include -o ${DIR_bin}/ftle ${FLAGS} ${OPENCL_FLAGS}
	cp ${DIR_src}/*.aocx ${DIR_bin}

compute_ftle_emu:
	${CC} ${DIR_src}/ftle.c ${SRC} -DEMULATION -I ./include -o ${DIR_bin}/ftle ${FLAGS} ${OPENCL_FLAGS}
	cp ${DIR_src}/*.aocx ${DIR_bin}

clean:
	rm -rf ${DIR_bin}/*
	#cd ${DIR_bin} && rm ${OBJS} && cd ..
