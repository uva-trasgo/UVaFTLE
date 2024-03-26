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

#include(CheckLanguage)
#check_language(CUDA)

#if(NOT CMAKE_CUDA_COMPILER)
#	message(FATAL_ERROR "Can't find CUDA")
#endif()

find_package(OpenMP)

if(NOT OpenMP_C_FOUND)
        message(FATAL_ERROR "Can't find OpenMP")
endif()


#enable_language(CUDA)

SET(CUDA_PATH "/opt/cuda-11.3")
SET(CFLAGS "-O3")
SET(OMP_FLAGS "-fopenmp")

SET(CPU_AND_CUDA_DIR "CPU/src")
SET(CPU_AND_CUDA_SRC ${CPU_AND_CUDA_DIR}/ftle.c ${CPU_AND_CUDA_DIR}/preprocess.c ${CPU_AND_CUDA_DIR}/arithmetic.c )
SET(CPU_AND_CUDA_FLAGS "${CFLAGS} ${OMP_FLAGS} -x cuda --cuda-gpu-arch=sm_70 -I./CPU/include -I${CUDA_PATH}/include -march=native")
SET(CUDA_LINK_FLAGS "-L${CUDA_PATH}/lib64 -fopenmp") 

#CPU versions with GPU for Preprocessing
ADD_EXECUTABLE(ftle_static  ${CPU_AND_CUDA_SRC})
ADD_EXECUTABLE(ftle_dynamic ${CPU_AND_CUDA_SRC})
ADD_EXECUTABLE(ftle_guided  ${CPU_AND_CUDA_SRC})

SET_TARGET_PROPERTIES(ftle_static PROPERTIES COMPILE_FLAGS ${CPU_AND_CUDA_FLAGS})
SET_TARGET_PROPERTIES(ftle_dynamic PROPERTIES COMPILE_FLAGS "${CPU_AND_CUDA_FLAGS} -DDYNAMIC")
SET_TARGET_PROPERTIES(ftle_guided  PROPERTIES COMPILE_FLAGS "${CPU_AND_CUDA_FLAGS} -DGUIDED")
SET_TARGET_PROPERTIES(ftle_guided ftle_dynamic ftle_static  PROPERTIES LINK_FLAGS "${CUDA_LINK_FLAGS}")



TARGET_LINK_LIBRARIES(ftle_static  m  cudart_static rt dl stdc++)
TARGET_LINK_LIBRARIES(ftle_dynamic m  cudart_static rt dl stdc++)
TARGET_LINK_LIBRARIES(ftle_guided  m  cudart_static rt dl stdc++)


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


TARGET_LINK_LIBRARIES(ftle_static_alone  m)
TARGET_LINK_LIBRARIES(ftle_dynamic_alone m)
TARGET_LINK_LIBRARIES(ftle_guided_alone  m)
SET_TARGET_PROPERTIES(ftle_guided_alone ftle_dynamic_alone ftle_static_alone  PROPERTIES LINK_FLAGS "-fopenmp")



#CUDA VERSIONS
SET(CUDA_DIR "CUDA/src")
SET(CUDA_SRC ${CUDA_DIR}/ftle.c  ${CUDA_DIR}/preprocess.c ${CUDA_DIR}/arithmetic.c)
SET(CUDA_FLAGS "${CFLAGS} ${OMP_FLAGS} -I./CUDA/include -x cuda --cuda-gpu-arch=sm_70 -I${CUDA_PATH}/include -march=native")


ADD_EXECUTABLE(ftle_cuda ${CUDA_SRC})
ADD_EXECUTABLE(ftle_cuda_pin ${CUDA_SRC})
SET_TARGET_PROPERTIES(ftle_cuda  PROPERTIES COMPILE_FLAGS ${CUDA_FLAGS})
SET_TARGET_PROPERTIES(ftle_cuda_pin  PROPERTIES COMPILE_FLAGS "${CUDA_FLAGS} -DPINNED")

SET_TARGET_PROPERTIES(ftle_cuda ftle_cuda_pin  PROPERTIES LINK_FLAGS ${CUDA_LINK_FLAGS})

IF(CUDA_ARCH)
	SET_TARGET_PROPERTIES(ftle_cuda  PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH})
	SET_TARGET_PROPERTIES(ftle_cuda_pin  PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH})
ENDIF()

TARGET_LINK_LIBRARIES(ftle_cuda m  cudart_static rt dl stdc++)
TARGET_LINK_LIBRARIES(ftle_cuda_pin m  cudart_static rt dl stdc++)

#ROCM
IF(NOT ROCM_DIR)
	SET(ROCM_DIR "/opt/rocm")
        message("-- SYCL with RoCM enabled but ROCM_DIR not specified . Setting by default to ${ROCM_DIR}")
ENDIF()

SET(HIP_DIR "HIP/src")
SET(HIP_SRC ${HIP_DIR}/ftle.c  ${HIP_DIR}/preprocess.c ${HIP_DIR}/arithmetic.c)
SET(HIP_FLAGS "${CFLAGS} ${OMP_FLAGS} -I./HIP/include -x hip --offload-arch=gfx900")


ADD_EXECUTABLE(ftle_hip ${HIP_SRC})
ADD_EXECUTABLE(ftle_hip_pin ${HIP_SRC})
SET(HIP_LINK_FLAGS "-L${ROCM_DIR}/lib -fopenmp")

SET_TARGET_PROPERTIES(ftle_hip ftle_hip_pin  PROPERTIES COMPILE_FLAGS ${HIP_FLAGS}) 
SET_TARGET_PROPERTIES(ftle_hip_pin  PROPERTIES COMPILE_FLAGS "${HIP_FLAGS} -DPINNED")
SET_TARGET_PROPERTIES(ftle_hip  ftle_hip_pin  PROPERTIES LINK_FLAGS "${HIP_LINK_FLAGS}") 

TARGET_LINK_LIBRARIES(ftle_hip m amdhip64)
TARGET_LINK_LIBRARIES(ftle_hip_pin m amdhip64)


INSTALL(TARGETS ftle_guided_alone ftle_dynamic_alone ftle_static_alone ftle_hip ftle_hip_pin ftle_cuda ftle_cuda_pin ftle_static ftle_dynamic ftle_guided  RUNTIME DESTINATION bin)




# SYCL Implementation

IF(WITH_SYCL_CPU STREQUAL "yes"  OR WITH_SYCL_CUDA STREQUAL "yes" OR WITH_SYCL_ROCM STREQUAL "yes" OR WITH_SYCL_GENERIC STREQUAL "yes")
	#NOTE FOR USERS: manually specify this vars editing CMakeLists.txt or
        #manually inicializate when cmake is invoked
	
	IF(NOT ADPCPP_DIR)
		message("-- SYCL enabled but ADPCCP_DIR  not specified. Setting by default to $ENV{HOME}/local/AdaptiveCPP")
		SET(ADPCPP_DIR "$ENV{HOME}/local/AdaptiveCPP")
	ENDIF()
	find_program(SYCL_CC "acpp")
	IF(NOT SYCL_CC)
		message(FATAL_ERROR "sycl compiler not found")
	ENDIF()
	set (CMAKE_CXX_COMPILER "${ADPCPP_DIR}/bin/acpp")
	set (CMAKE_CXX_LINKER "${ADPCPP_DIR}/bin/acpp")
	

	#NOTE FOR USERS: manually specify this vars editing CMakeLists.txt or
	#manually inicializate when cmake is invoked
	IF(NOT BOOST_DIR)
		message("-- SYCL enabled but BOOST_DIR not specified (required for CUDA and ROCM). Setting by default to $ENV{HOME}/local/boost")
		SET(BOOST_DIR "$ENV{HOME}/local/boost")
	ENDIF()

	SET(SYCL_SRCDIR "SYCL/src")
	SET(USM_SRCDIR "SYCL-usm/src")
	SET(SYCL_SRC  ${SYCL_SRCDIR}/preprocess.cpp ${SYCL_SRCDIR}/arithmetic.cpp ${SYCL_SRCDIR}/ftle.cpp)
	SET(USM_SRC  ${USM_SRCDIR}/preprocess.cpp ${USM_SRCDIR}/arithmetic.cpp 	${USM_SRCDIR}/ftle.cpp)
	SET(SPLIT_SRC  ${USM_SRCDIR}/preprocess.cpp ${USM_SRCDIR}/arithmetic.cpp 	${USM_SRCDIR}/ftle-split.cpp)

	IF(WITH_SYCL_CPU STREQUAL "yes")
		SET(SYCL_COMPILE_FLAGS "${CFLAGS} -march=native --acpp-targets='omp'  -DBLOCK=512")
		SET(SYCL_LINK_FLAGS "-lstdc++ -L${BOOST_DIR}/lib --acpp-targets='omp'  -lstdc++ -L${BOOST_DIR}/lib  -lm")
		ADD_EXECUTABLE(ftle_sycl_cpu  ${SYCL_SRC})
		ADD_EXECUTABLE(ftle_usm_cpu ${USM_SRC})
		SET_TARGET_PROPERTIES(ftle_sycl_cpu PROPERTIES COMPILE_FLAGS "${SYCL_COMPILE_FLAGS} -I./SYCL/include")
		SET_TARGET_PROPERTIES(ftle_usm_cpu PROPERTIES COMPILE_FLAGS "${SYCL_COMPILE_FLAGS} -I./SYCL-usm/include")
		SET_TARGET_PROPERTIES(ftle_sycl_cpu ftle_usm_cpu  PROPERTIES LINK_FLAGS ${SYCL_LINK_FLAGS})
		INSTALL(TARGETS ftle_sycl_cpu ftle_usm_cpu RUNTIME DESTINATION bin)	
	ENDIF()
	
	#NOTE FOR USERS: Change the architecture sm_70 for other nvidia devices
	IF(WITH_SYCL_CUDA STREQUAL "yes")		
		SET(SYCL_COMPILE_FLAGS "${CFLAGS}  --acpp-targets='cuda:sm_70' -DCUDA_DEVICE -DBLOCK=512")
		SET(SYCL_LINK_FLAGS " -L${BOOST_DIR}/lib --acpp-targets='cuda:sm_70'")
		ADD_EXECUTABLE(ftle_sycl_cuda  ${SYCL_SRC})
		ADD_EXECUTABLE(ftle_usm_cuda ${USM_SRC})
		ADD_EXECUTABLE(ftle_usm_split_cuda ${SPLIT_SRC})
		SET_TARGET_PROPERTIES(ftle_sycl_cuda PROPERTIES COMPILE_FLAGS "${SYCL_COMPILE_FLAGS} -I./SYCL/include")
		SET_TARGET_PROPERTIES(ftle_usm_cuda ftle_usm_split_cuda PROPERTIES COMPILE_FLAGS "${SYCL_COMPILE_FLAGS} -I./SYCL-usm/include")
		SET_TARGET_PROPERTIES(ftle_sycl_cuda ftle_usm_cuda ftle_usm_split_cuda PROPERTIES LINK_FLAGS ${SYCL_LINK_FLAGS})
		INSTALL(TARGETS ftle_sycl_cuda ftle_usm_cuda ftle_usm_split_cuda RUNTIME DESTINATION bin)
	ENDIF()
	
	#NOTE FOR USERS: Change the architecture gfx900 for other amd devices
	IF(WITH_SYCL_ROCM STREQUAL "yes")
		SET(ROCM_COMMON_FLAGS "--rocm-path=${ROCM_DIR} --acpp-targets='hip:gfx900' -I${ROCM_DIR}/include -DHIP_DEVICE  -DBLOCK=512")
		SET(SYCL_COMPILE_FLAGS "${CFLAGS} ${ROCM_COMMON_FLAGS}")
		SET(SYCL_LINK_FLAGS "-lstdc++ -L${ROCM_DIR}/lib -L${ROCM_DIR}/lib64 -L${BOOST_DIR}/lib ${ROCM_COMMON_FLAGS} -lm")
		ADD_EXECUTABLE(ftle_sycl_rocm  ${SYCL_SRC})
		ADD_EXECUTABLE(ftle_usm_rocm  ${USM_SRC})
		ADD_EXECUTABLE(ftle_usm_split_rocm ${SPLIT_SRC})
		SET_TARGET_PROPERTIES(ftle_sycl_rocm PROPERTIES COMPILE_FLAGS "${SYCL_COMPILE_FLAGS} -I./SYCL/include")
		SET_TARGET_PROPERTIES(ftle_usm_rocm ftle_usm_split_rocm PROPERTIES COMPILE_FLAGS "${SYCL_COMPILE_FLAGS} -I./SYCL-usm/include")
		SET_TARGET_PROPERTIES(ftle_sycl_rocm ftle_usm_rocm ftle_usm_split_rocm PROPERTIES LINK_FLAGS ${SYCL_LINK_FLAGS})
		INSTALL(TARGETS ftle_sycl_rocm ftle_usm_rocm ftle_usm_split_rocm RUNTIME DESTINATION bin)
	ENDIF()
	
	IF(WITH_SYCL_CUDA STREQUAL "yes" AND WITH_SYCL_ROCM STREQUAL "yes")
	   SET(TARGETS "hip:gfx900;cuda:sm_70")
		SET(SYCL_COMPILE_FLAGS "${CFLAGS} --acpp-explicit-multipass -DGPU_ALL  -I${ROCM_DIR}/include/  -DBLOCK=512")
		SET(SYCL_LINK_FLAGS "-lstdc++ --acpp-explicit-multipass -L${BOOST_DIR}/lib  -L${ROCM_DIR}/lib -L${ROCM_DIR}/lib64 -L${BOOST_DIR}/lib ${ROCM_COMMON_FLAGS} -lm")
		ADD_EXECUTABLE(ftle_sycl_gpu  ${SYCL_SRC})
		#ADD_EXECUTABLE(ftle_usm_gpu  ${USM_SRC})
		#ADD_EXECUTABLE(ftle_usm_split_gpu  ${SPLIT_SRC})
		SET_TARGET_PROPERTIES(ftle_sycl_gpu PROPERTIES COMPILE_FLAGS "${SYCL_COMPILE_FLAGS} -I./SYCL/include --acpp-targets='${TARGETS}'" )
		#SET_TARGET_PROPERTIES(ftle_usm_split_gpu PROPERTIES COMPILE_FLAGS "${SYCL_COMPILE_FLAGS} -I./SYCL-usm/include --acpp-targets='${TARGETS}'")
		SET_TARGET_PROPERTIES(ftle_sycl_gpu PROPERTIES LINK_FLAGS "${SYCL_LINK_FLAGS} --acpp-targets='${TARGETS}'")
		INSTALL(TARGETS ftle_sycl_gpu  RUNTIME DESTINATION bin)

	ENDIF()

	IF(WITH_SYCL_GENERIC STREQUAL "yes")
		SET(SYCL_COMPILE_FLAGS "${CFLAGS} -DGPU_ALL --acpp-targets='generic'  -I${ROCM_DIR}/hip/include/ -DBLOCK=512")
		SET(SYCL_LINK_FLAGS "-lstdc++ -L${BOOST_DIR}/lib --acpp-targets='generic'")
		ADD_EXECUTABLE(ftle_sycl_generic  ${SYCL_SRC})
		ADD_EXECUTABLE(ftle_usm_generic  ${USM_SRC})
		ADD_EXECUTABLE(ftle_usm_split_generic  ${SPLIT_SRC})
		SET_TARGET_PROPERTIES(ftle_sycl_generic PROPERTIES COMPILE_FLAGS "${SYCL_COMPILE_FLAGS} -I./SYCL/include")
		SET_TARGET_PROPERTIES(ftle_usm_generic ftle_usm_split_generic PROPERTIES COMPILE_FLAGS "${SYCL_COMPILE_FLAGS} -I./SYCL-usm/include")
		SET_TARGET_PROPERTIES(ftle_sycl_generic ftle_usm_generic ftle_usm_split_generic PROPERTIES LINK_FLAGS ${SYCL_LINK_FLAGS})
		INSTALL(TARGETS ftle_sycl_generic ftle_usm_generic ftle_usm_split_generic RUNTIME DESTINATION bin)
	ENDIF()
ENDIF()
