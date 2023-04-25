# Compilers
CC=clang++

# Flags
FLAGS=-O3 -lm 
FLAG_OMP= -fopenmp -march=native

# Directories
DIR=.
DIR_src=${DIR}/src
DIR_bin=${DIR}/bin

# Complementary files
SRC=${DIR_src}/preprocess.c ${DIR_src}/arithmetic.c

# Make lists
all: compute_ftle

# -------------------------- #
# ---------- GCC ----------- #
# -------------------------- #

compute_ftle:
	${CC} ${DIR_src}/ftle.c ${SRC} ${FLAG_OMP} -I ./include -o ${DIR_bin}/ftle_static ${FLAGS}
	${CC} ${DIR_src}/ftle.c ${SRC} ${FLAG_OMP} -DDYNAMIC -I ./include -o ${DIR_bin}/ftle_dynamic ${FLAGS}
	${CC} ${DIR_src}/ftle.c ${SRC} ${FLAG_OMP} -DGUIDED -I ./include -o ${DIR_bin}/ftle_guided ${FLAGS}

clean:
	cd ${DIR_bin} && rm ${OBJS} && cd ..
