cmake_minimum_required (VERSION 3.8)


PROJECT(UVaFTLE C CXX)

set (CMAKE_C_COMPILER "nvcc")

#SET DEFAULT INSTALL DIR
if (CMAKE_INSTALL_PREFIX STREQUAL "/usr/local")
	SET(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR})
endif()

include(CheckLanguage)
check_language(CUDA)

if(NOT CMAKE_CUDA_COMPILER)
	message(FATAL_ERROR "Can't find CUDA")
endif()

find_package(OpenMP)

if(NOT OpenMP_C_FOUND)
        message(FATAL_ERROR "Can't find OpenMP")
endif()


enable_language(CUDA)

if(${CMAKE_VERSION} VERSION_GREATER "3.17") 
	IF(NOT CUDA_ARCH)
		message("CUDA_ARCH not specified. Compiling for 75 architecture")
		SET(CUDA_ARCH "75")
	ELSE()
		message("Compiling for ${CUDA_ARCH} architecture(s)")
	ENDIF()
ELSE()
	UNSET(CUDA_ARCH)
ENDIF()

SET(CFLAGS "-O3")
SET(ZEN3_FLAGS "-march=znver3")
SET(OMP_FLAGS "-Xcompiler -fopenmp")

SET(CPU_DIR "CPU/src")
SET(CPU_SRC ${CPU_DIR}/ftle.cu ${CPU_DIR}/preprocess.cu ${CPU_DIR}/arithmetic.cu)
SET(CPU_FLAGS "${CFLAGS} ${OMP_FLAGS} -I./CPU/include -allow-unsupported-compiler")
 
#CPU versions
ADD_EXECUTABLE(ftle_static  ${CPU_SRC})
ADD_EXECUTABLE(ftle_dynamic ${CPU_SRC})
ADD_EXECUTABLE(ftle_guided  ${CPU_SRC})

SET_TARGET_PROPERTIES(ftle_static PROPERTIES COMPILE_FLAGS ${CPU_FLAGS})
SET_TARGET_PROPERTIES(ftle_dynamic PROPERTIES COMPILE_FLAGS "${CPU_FLAGS} -DDYNAMIC")
SET_TARGET_PROPERTIES(ftle_guided  PROPERTIES COMPILE_FLAGS "${CPU_FLAGS} -DGUIDED")

IF(CUDA_ARCH)
	SET_TARGET_PROPERTIES(ftle_static ftle_dynamic ftle_guided  PROPERTIES CUDA_ARCHITECTURES OFF)
ENDIF()


TARGET_LINK_LIBRARIES(ftle_static  m OpenMP::OpenMP_C)
TARGET_LINK_LIBRARIES(ftle_dynamic m OpenMP::OpenMP_C)
TARGET_LINK_LIBRARIES(ftle_guided  m OpenMP::OpenMP_C)

#CUDA VERSIONS
SET(CUDA_DIR "CUDA/src")
SET(CUDA_COMMON_SRC ${CUDA_DIR}/preprocess.cu ${CUDA_DIR}/arithmetic.cu)
SET(CUDA_FLAGS "${CFLAGS} ${OMP_FLAGS} -I./CUDA/include -allow-unsupported-compiler")


ADD_EXECUTABLE(ftle_cuda ${CUDA_DIR}/ftle.cu ${CUDA_COMMON_SRC})
SET_TARGET_PROPERTIES(ftle_cuda  PROPERTIES COMPILE_FLAGS ${CUDA_FLAGS})

IF(CUDA_ARCH)
	SET_TARGET_PROPERTIES(ftle_cuda  PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH})
ENDIF()

TARGET_LINK_LIBRARIES(ftle_cuda m OpenMP::OpenMP_C)


# SYCL Implementation

IF(WITH_SYCL_CPU STREQUAL "yes"  OR WITH_SYCL_CUDA STREQUAL "yes" OR WITH_SYCL_ROCM STREQUAL "yes")
	set(CMAKE_VERBOSE_MAKEFILE ON)
	find_program(SYCL_CC "syclcc")
	IF(NOT SYCL_CC)
		message(FATAL_ERROR "sycl compiler not found")
	ENDIF()
	set (CMAKE_CXX_COMPILER "syclcc")
	set (CMAKE_CXX_LINKER "syclcc")
	

	#NOTE FOR USERS: manually specify this vars editing CMakeLists.txt or
	#manually inicializate when cmake is invoked
	IF(NOT BOOST_DIR)
		message("SYCL enabled but BOOST_DIR not specified (required for CUDA and ROCM). Setting by default to $ENV{HOME}/local/boost")
		SET(BOOST_DIR "$ENV{HOME}/local/boost")
	ENDIF()

        IF(NOT ROCM_DIR AND WITH_SYCL_ROCM STREQUAL "yes")
                SET(ROCM_DIR "/opt/rocm-4.2.0")
                message("SYCL with RoCM enabled but ROCM_DIR not specified . Setting by default to ${ROCM_DIR}")
        ENDIF()



	SET(SYCL_SRCDIR "SYCL/src")
	SET(SYCL_SRC  ${SYCL_SRCDIR}/preprocess.cpp
		${SYCL_SRCDIR}/arithmetic.cpp ${SYCL_SRCDIR}/ftle.cpp)

	IF(WITH_SYCL_CPU STREQUAL "yes")
                SET(SYCL_COMPILE_FLAGS "${CFLAGS} -I./SYCL/include --hipsycl-targets='omp\;cuda:sm_60' --hipsycl-cpu-cxx=/home/trasgo/local/llvm14/bin/clang++")
                SET(SYCL_LINK_FLAGS "-lstdc++ -L${BOOST_DIR}/lib --hipsycl-targets='omp\;cuda:sm_60' --hipsycl-cpu-cxx=/home/trasgo/local/llvm14/bin/clang++ -lstdc++ -L${BOOST_DIR}/lib  -lm")
                ADD_EXECUTABLE(ftle_sycl_cpu  ${SYCL_SRC})
                SET_TARGET_PROPERTIES(ftle_sycl_cpu  PROPERTIES COMPILE_FLAGS  ${SYCL_COMPILE_FLAGS})
                SET_TARGET_PROPERTIES(ftle_sycl_cpu  PROPERTIES LINK_FLAGS ${SYCL_LINK_FLAGS})
                INSTALL(TARGETS ftle_sycl_cpu RUNTIME DESTINATION bin)

	ENDIF()
	
	IF(WITH_SYCL_CUDA STREQUAL "yes")
		#TODO configurar arquitectura en CmakeLIST
		SET(SYCL_COMPILE_FLAGS "${CFLAGS} -I./SYCL/include --hipsycl-targets=cuda:sm_60 -DCUDA_DEVICE")
		SET(SYCL_LINK_FLAGS "-lstdc++ -L${BOOST_DIR}/lib --hipsycl-targets=cuda:sm_60")
		ADD_EXECUTABLE(ftle_sycl_cuda  ${SYCL_SRC})
		SET_TARGET_PROPERTIES(ftle_sycl_cuda  PROPERTIES COMPILE_FLAGS 	${SYCL_COMPILE_FLAGS})
		SET_TARGET_PROPERTIES(ftle_sycl_cuda  PROPERTIES LINK_FLAGS ${SYCL_LINK_FLAGS})
		INSTALL(TARGETS ftle_sycl_cuda RUNTIME DESTINATION bin)
	ENDIF()

	IF(WITH_SYCL_ROCM STREQUAL "yes")
		#TODO configurar arquitectura en CmakeLIST
		SET(ROCM_COMMON_FLAGS "--rocm-path=${ROCM_DIR} --hipsycl-targets=hip:gfx900  -I/opt/rocm-4.2.0/hip/include/hip/ -DHIP_DEVICE")
		SET(SYCL_COMPILE_FLAGS "${CFLAGS} -I./SYCL/include ${ROCM_COMMON_FLAGS}")
		SET(SYCL_LINK_FLAGS "-lstdc++ -L${ROCM_DIR}/lib -L${ROCM_DIR}/lib64 -L${BOOST_DIR}/lib ${ROCM_COMMON_FLAGS} -lm")
		ADD_EXECUTABLE(ftle_sycl_rocm  ${SYCL_SRC})
		SET_TARGET_PROPERTIES(ftle_sycl_rocm  PROPERTIES COMPILE_FLAGS 	${SYCL_COMPILE_FLAGS})
		SET_TARGET_PROPERTIES(ftle_sycl_rocm  PROPERTIES LINK_FLAGS ${SYCL_LINK_FLAGS})
		INSTALL(TARGETS ftle_sycl_rocm RUNTIME DESTINATION bin)
	ENDIF()
	
	IF(WITH_SYCL_CUDA STREQUAL "yes" AND WITH_SYCL_ROCM STREQUAL "yes")
           SET(SYCL_COMPILE_FLAGS "${CFLAGS} -I./SYCL/include -DGPU_ALL --hipsycl-explicit-multipass --hipsycl-targets='omp\;hip:gfx900\;cuda:sm_60' --hipsycl-cpu-cxx=/home/trasgo/local/llvm14/bin/clang++ --rocm-path=${ROCM_DIR}  -I/opt/rocm-4.2.0/hip/include/")
           SET(SYCL_LINK_FLAGS "-lstdc++ -L${BOOST_DIR}/lib --hipsycl-explicit-multipass --hipsycl-targets='omp\;hip:gfx900\;cuda:sm_60' --hipsycl-cpu-cxx=/home/trasgo/local/llvm14/bin/clang++ -lstdc++ -L${ROCM_DIR}/lib -L${ROCM_DIR}/lib64 -L${BOOST_DIR}/lib ${ROCM_COMMON_FLAGS} -lm")
           ADD_EXECUTABLE(ftle_sycl_gpu  ${SYCL_SRC})
           SET_TARGET_PROPERTIES(ftle_sycl_gpu  PROPERTIES COMPILE_FLAGS  ${SYCL_COMPILE_FLAGS})
           SET_TARGET_PROPERTIES(ftle_sycl_gpu  PROPERTIES LINK_FLAGS ${SYCL_LINK_FLAGS})
           INSTALL(TARGETS ftle_sycl_gpu RUNTIME DESTINATION bin)

	ENDIF()
	
ENDIF()

INSTALL(TARGETS ftle_cuda ftle_static ftle_dynamic ftle_guided  RUNTIME DESTINATION bin)
