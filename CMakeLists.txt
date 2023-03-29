cmake_minimum_required (VERSION 3.8)


PROJECT(UVaFTLE C CXX)

set (CMAKE_C_COMPILER "clang++")
set(CMAKE_VERBOSE_MAKEFILE ON)
#SET DEFAULT INSTALL DIR
if (CMAKE_INSTALL_PREFIX STREQUAL "/usr/local")
	SET(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR})
endif()


#check if the symbolink links exists
#if(NOT EXISTS "CUDA/src/ftle.c")
	execute_process(COMMAND ln -s "$ENV{PWD}/CUDA/src/ftle.cu" "$ENV{PWD}/CUDA/src/ftle.c" OUTPUT_QUIET ERROR_QUIET)
	execute_process(COMMAND ln -s "$ENV{PWD}/CUDA/src/arithmetic.cu" "$ENV{PWD}/CUDA/src/arithmetic.c"  OUTPUT_QUIET ERROR_QUIET)
	execute_process(COMMAND ln -s "$ENV{PWD}/CUDA/src/preprocess.cu" "$ENV{PWD}/CUDA/src/preprocess.c"  OUTPUT_QUIET ERROR_QUIET)
	execute_process(COMMAND ln -s "$ENV{PWD}/CPU/src/ftle.cu" "$ENV{PWD}/CPU/src/ftle.c" OUTPUT_QUIET ERROR_QUIET)
	execute_process(COMMAND ln -s "$ENV{PWD}/CPU/src/arithmetic.cu" "$ENV{PWD}/CPU/src/arithmetic.c"  OUTPUT_QUIET ERROR_QUIET)
	execute_process(COMMAND ln -s "$ENV{PWD}/CPU/src/preprocess.cu" "$ENV{PWD}/CPU/src/preprocess.c" OUTPUT_QUIET ERROR_QUIET)
	execute_process(COMMAND ln -s "$ENV{PWD}/HIP/src/ftle.cpp" "$ENV{PWD}/HIP/src/ftle.c" OUTPUT_QUIET ERROR_QUIET)
	execute_process(COMMAND ln -s "$ENV{PWD}/HIP/src/arithmetic.cpp" "$ENV{PWD}/HIP/src/arithmetic.c" OUTPUT_QUIET ERROR_QUIET)
	execute_process(COMMAND ln -s "$ENV{PWD}/HIP/src/preprocess.cpp" "$ENV{PWD}/HIP/src/preprocess.c" OUTPUT_QUIET ERROR_QUIET)

#endif() 

include(CheckLanguage)
check_language(CUDA)

if(NOT CMAKE_CUDA_COMPILER)
	message(FATAL_ERROR "Can't find CUDA")
endif()

find_package(OpenMP)

if(NOT OpenMP_C_FOUND)
        message(FATAL_ERROR "Can't find OpenMP")
endif()


#enable_language(CUDA)

#if(${CMAKE_VERSION} VERSION_GREATER "3.17") 
#	IF(NOT CUDA_ARCH)
#		message("CUDA_ARCH not specified. Compiling for 70 architecture")
#		SET(CUDA_ARCH "70")
#	ELSE()
#		message("Compiling for ${CUDA_ARCH} architecture(s)")
#	ENDIF()
#ELSE()
#	UNSET(CUDA_ARCH)
#ENDIF()
SET(CUDA_PATH "/opt/cuda-11.3")
SET(CFLAGS "-O3")
SET(OMP_FLAGS "-fopenmp")

SET(CPU_AND_CUDA_DIR "CPU/src")
SET(CPU_AND_CUDA_SRC ${CPU_AND_CUDA_DIR}/ftle.c ${CPU_AND_CUDA_DIR}/preprocess.c ${CPU_AND_CUDA_DIR}/arithmetic.c )
SET(CPU_AND_CUDA_FLAGS "${CFLAGS} ${OMP_FLAGS} -x cuda --cuda-gpu-arch=sm_70 -I./CPU/include -I${CUDA_PATH}/include -march=native")
SET(CUDA_LINK_FLAGS "-L${CUDA_PATH}/lib64") 

#CPU versions with GPU for Preprocessing
ADD_EXECUTABLE(ftle_static  ${CPU_AND_CUDA_SRC})
ADD_EXECUTABLE(ftle_dynamic ${CPU_AND_CUDA_SRC})
ADD_EXECUTABLE(ftle_guided  ${CPU_AND_CUDA_SRC})

SET_TARGET_PROPERTIES(ftle_static PROPERTIES COMPILE_FLAGS ${CPU_AND_CUDA_FLAGS})
SET_TARGET_PROPERTIES(ftle_dynamic PROPERTIES COMPILE_FLAGS "${CPU_AND_CUDA_FLAGS} -DDYNAMIC")
SET_TARGET_PROPERTIES(ftle_guided  PROPERTIES COMPILE_FLAGS "${CPU_AND_CUDA_FLAGS} -DGUIDED")
SET_TARGET_PROPERTIES(ftle_guided ftle_dynamic ftle_static  PROPERTIES LINK_FLAGS ${CUDA_LINK_FLAGS})



TARGET_LINK_LIBRARIES(ftle_static  m OpenMP::OpenMP_C cudart_static rt dl stdc++)
TARGET_LINK_LIBRARIES(ftle_dynamic m OpenMP::OpenMP_C cudart_static rt dl stdc++)
TARGET_LINK_LIBRARIES(ftle_guided  m OpenMP::OpenMP_C cudart_static rt dl stdc++)

#CPU ALONE versions 
SET(CPU_DIR "CPU-alone/src")
SET(CPU_SRC ${CPU_DIR}/ftle.c ${CPU_DIR}/preprocess.c ${CPU_DIR}/arithmetic.c )
SET(CPU_FLAGS "${CFLAGS} ${OMP_FLAGS} -I./CPU-alone/include -march=native")

ADD_EXECUTABLE(ftle_static_alone  ${CPU_SRC})
ADD_EXECUTABLE(ftle_dynamic_alone ${CPU_SRC})
ADD_EXECUTABLE(ftle_guided_alone  ${CPU_SRC})

SET_TARGET_PROPERTIES(ftle_static_alone PROPERTIES COMPILE_FLAGS ${CPU_FLAGS})
SET_TARGET_PROPERTIES(ftle_dynamic_alone PROPERTIES COMPILE_FLAGS "${CPU_FLAGS} -DDYNAMIC")
SET_TARGET_PROPERTIES(ftle_guided_alone  PROPERTIES COMPILE_FLAGS "${CPU_FLAGS} -DGUIDED")

IF(CUDA_ARCH)
	SET_TARGET_PROPERTIES(ftle_static ftle_dynamic ftle_guided  PROPERTIES CUDA_ARCHITECTURES OFF)
ENDIF()


TARGET_LINK_LIBRARIES(ftle_static  m OpenMP::OpenMP_C)
TARGET_LINK_LIBRARIES(ftle_dynamic m OpenMP::OpenMP_C)
TARGET_LINK_LIBRARIES(ftle_guided  m OpenMP::OpenMP_C)


#CUDA VERSIONS
SET(CUDA_DIR "CUDA/src")
SET(CUDA_SRC ${CUDA_DIR}/ftle.c  ${CUDA_DIR}/preprocess.c ${CUDA_DIR}/arithmetic.c)
SET(CUDA_FLAGS "${CFLAGS} ${OMP_FLAGS} -I./CUDA/include -x cuda --cuda-gpu-arch=sm_70 -I${CUDA_PATH}/include -march=native")


ADD_EXECUTABLE(ftle_cuda ${CUDA_SRC})
ADD_EXECUTABLE(ftle_cuda_pin ${CUDA_SRC})
SET_TARGET_PROPERTIES(ftle_cuda  PROPERTIES COMPILE_FLAGS ${CUDA_FLAGS})
SET_TARGET_PROPERTIES(ftle_cuda_pin  PROPERTIES COMPILE_FLAGS "${CUDA_FLAGS} -DPINNED")

SET_TARGET_PROPERTIES(ftle_guided ftle_dynamic ftle_static ftle_cuda ftle_cuda_pin  PROPERTIES LINK_FLAGS ${CUDA_LINK_FLAGS})
IF(CUDA_ARCH)
	SET_TARGET_PROPERTIES(ftle_cuda  PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH})
	SET_TARGET_PROPERTIES(ftle_cuda_pin  PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH})
ENDIF()

TARGET_LINK_LIBRARIES(ftle_cuda m OpenMP::OpenMP_C cudart_static rt dl stdc++)
TARGET_LINK_LIBRARIES(ftle_cuda_pin m OpenMP::OpenMP_C cudart_static rt dl stdc++)

#ROCM
IF(NOT ROCM_DIR)
	SET(ROCM_DIR "/opt/rocm")
        message("-- SYCL with RoCM enabled but ROCM_DIR not specified . Setting by default to ${ROCM_DIR}")
ENDIF()

SET(HIP_DIR "HIP/src")
SET(HIP_SRC ${HIP_DIR}/ftle.c  ${HIP_DIR}/preprocess.c ${HIP_DIR}/arithmetic.c)
SET(HIP_FLAGS "${CFLAGS} ${OMP_FLAGS} -I./HIP/include -x hip --offload-arch=gfx900 -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/oclc_abi_version_400.bc")


ADD_EXECUTABLE(ftle_hip ${HIP_SRC})
ADD_EXECUTABLE(ftle_hip_pin ${HIP_SRC})
SET(HIP_LINK_FLAGS "-L${ROCM_DIR}/lib ")

SET_TARGET_PROPERTIES(ftle_hip ftle_hip_pin  PROPERTIES COMPILE_FLAGS ${HIP_FLAGS}) 
SET_TARGET_PROPERTIES(ftle_hip_pin  PROPERTIES COMPILE_FLAGS "${HIP_FLAGS} -DPINNED")
SET_TARGET_PROPERTIES(ftle_hip  ftle_hip_pin  PROPERTIES LINK_FLAGS "${HIP_LINK_FLAGS}") 

TARGET_LINK_LIBRARIES(ftle_hip m OpenMP::OpenMP_C amdhip64)
TARGET_LINK_LIBRARIES(ftle_hip_pin m OpenMP::OpenMP_C amdhip64)


INSTALL(TARGETS ftle_hip ftle_hip_pin ftle_cuda ftle_cuda_pin ftle_static ftle_dynamic ftle_guided  RUNTIME DESTINATION bin)




# SYCL Implementation

IF(WITH_SYCL_CPU STREQUAL "yes"  OR WITH_SYCL_CUDA STREQUAL "yes" OR WITH_SYCL_ROCM STREQUAL "yes")
	SET(OPENSYCL_DIR "/home/trasgo/local/openSYCL")
	find_program(SYCL_CC "syclcc")
	IF(NOT SYCL_CC)
		message(FATAL_ERROR "sycl compiler not found")
	ENDIF()
	set (CMAKE_CXX_COMPILER "${OPENSYCL_DIR}/bin/syclcc")
	set (CMAKE_CXX_LINKER "${OPENSYCL_DIR}/bin/syclcc")
	

	#NOTE FOR USERS: manually specify this vars editing CMakeLists.txt or
	#manually inicializate when cmake is invoked
	IF(NOT BOOST_DIR)
		message("-- SYCL enabled but BOOST_DIR not specified (required for CUDA and ROCM). Setting by default to $ENV{HOME}/local/boost")
		SET(BOOST_DIR "$ENV{HOME}/local/boost")
	ENDIF()





	SET(SYCL_SRCDIR "SYCL/src")
	SET(SYCL_SRC  ${SYCL_SRCDIR}/preprocess.cpp
		${SYCL_SRCDIR}/arithmetic.cpp ${SYCL_SRCDIR}/ftle.cpp)

	IF(WITH_SYCL_CPU STREQUAL "yes")
                SET(SYCL_COMPILE_FLAGS "${CFLAGS} -I./SYCL/include -march=native --hipsycl-targets='omp' --hipsycl-cpu-cxx=/home/trasgo/local/llvm14/bin/clang++")
                SET(SYCL_LINK_FLAGS "-lstdc++ -L${BOOST_DIR}/lib --hipsycl-targets='omp' --hipsycl-cpu-cxx=/home/trasgo/local/llvm14/bin/clang++ -lstdc++ -L${BOOST_DIR}/lib  -lm")
                ADD_EXECUTABLE(ftle_sycl_cpu  ${SYCL_SRC})
                SET_TARGET_PROPERTIES(ftle_sycl_cpu  PROPERTIES COMPILE_FLAGS  ${SYCL_COMPILE_FLAGS})
                SET_TARGET_PROPERTIES(ftle_sycl_cpu  PROPERTIES LINK_FLAGS ${SYCL_LINK_FLAGS})
                INSTALL(TARGETS ftle_sycl_cpu RUNTIME DESTINATION bin)

	ENDIF()
	
	SET (ARCH "sm_70" )
	IF(WITH_SYCL_CUDA STREQUAL "yes")
		#TODO configurar arquitectura en CmakeLIST
		SET(SYCL_COMPILE_FLAGS "${CFLAGS} -I./SYCL/include --hipsycl-targets=cuda:sm_70 -DCUDA_DEVICE")
		SET(SYCL_LINK_FLAGS " -L${BOOST_DIR}/lib --hipsycl-targets=cuda:sm_70")
		ADD_EXECUTABLE(ftle_sycl_cuda  ${SYCL_SRC})
		SET_TARGET_PROPERTIES(ftle_sycl_cuda  PROPERTIES COMPILE_FLAGS 	${SYCL_COMPILE_FLAGS})
		SET_TARGET_PROPERTIES(ftle_sycl_cuda  PROPERTIES LINK_FLAGS ${SYCL_LINK_FLAGS})
		INSTALL(TARGETS ftle_sycl_cuda RUNTIME DESTINATION bin)
	ENDIF()
	
	IF(WITH_SYCL_ROCM STREQUAL "yes")
		#TODO configurar arquitectura en CmakeLIST
		SET(ROCM_COMMON_FLAGS "--rocm-path=${ROCM_DIR} --hipsycl-targets=hip:gfx900 -I${ROCM_DIR}/hip/include/hip/ -DHIP_DEVICE -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/oclc_abi_version_400.bc")
		SET(SYCL_COMPILE_FLAGS "${CFLAGS} -I./SYCL/include ${ROCM_COMMON_FLAGS}")
		SET(SYCL_LINK_FLAGS "-lstdc++ -L${ROCM_DIR}/lib -L${ROCM_DIR}/lib64 -L${BOOST_DIR}/lib ${ROCM_COMMON_FLAGS} -lm")
		ADD_EXECUTABLE(ftle_sycl_rocm  ${SYCL_SRC})
		SET_TARGET_PROPERTIES(ftle_sycl_rocm  PROPERTIES COMPILE_FLAGS 	${SYCL_COMPILE_FLAGS})
		SET_TARGET_PROPERTIES(ftle_sycl_rocm  PROPERTIES LINK_FLAGS ${SYCL_LINK_FLAGS})
		INSTALL(TARGETS ftle_sycl_rocm RUNTIME DESTINATION bin)
	ENDIF()
	
	IF(WITH_SYCL_CUDA STREQUAL "yes" AND WITH_SYCL_ROCM STREQUAL "yes")
           SET(SYCL_COMPILE_FLAGS "${CFLAGS} -I./SYCL/include -DGPU_ALL --hipsycl-explicit-multipass --hipsycl-targets='hip:gfx900\;cuda:sm_70' --hipsycl-cpu-cxx=/home/trasgo/local/llvm14/bin/clang++ --rocm-path=${ROCM_DIR} -I${ROCM_DIR}/hip/include/ -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/oclc_abi_version_400.bc")
           SET(SYCL_LINK_FLAGS "-lstdc++ -L${BOOST_DIR}/lib --hipsycl-explicit-multipass --hipsycl-targets='hip:gfx900\;cuda:sm_70' --hipsycl-cpu-cxx=/home/trasgo/local/llvm14/bin/clang++ -lstdc++ -L${ROCM_DIR}/lib -L${ROCM_DIR}/lib64 -L${BOOST_DIR}/lib ${ROCM_COMMON_FLAGS} -lm")
           ADD_EXECUTABLE(ftle_sycl_gpu  ${SYCL_SRC})
           SET_TARGET_PROPERTIES(ftle_sycl_gpu  PROPERTIES COMPILE_FLAGS  ${SYCL_COMPILE_FLAGS})
           SET_TARGET_PROPERTIES(ftle_sycl_gpu  PROPERTIES LINK_FLAGS ${SYCL_LINK_FLAGS})
           INSTALL(TARGETS ftle_sycl_gpu RUNTIME DESTINATION bin)

	ENDIF()
	
ENDIF()
