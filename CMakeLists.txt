cmake_minimum_required (VERSION 3.8)


PROJECT(UVaFTLE C)

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
SET(CPU_FLAGS "${CFLAGS} ${OMP_FLAGS} -I./CPU/include")
 
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
SET(CUDA_FLAGS "${CFLAGS} ${OMP_FLAGS} -I./CUDA/include")


ADD_EXECUTABLE(ftle_cuda ${CUDA_DIR}/ftle.cu ${CUDA_COMMON_SRC})
ADD_EXECUTABLE(ftle_cuda_pin ${CUDA_DIR}/ftle.cu ${CUDA_COMMON_SRC})
SET_TARGET_PROPERTIES(ftle_cuda  PROPERTIES COMPILE_FLAGS ${CUDA_FLAGS})
SET_TARGET_PROPERTIES(ftle_cuda_pin  PROPERTIES COMPILE_FLAGS ${CUDA_FLAGS} -DPINNED)

IF(CUDA_ARCH)
	SET_TARGET_PROPERTIES(ftle_cuda  PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH})
	SET_TARGET_PROPERTIES(ftle_cuda_pin  PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH})
ENDIF()

TARGET_LINK_LIBRARIES(ftle_cuda m OpenMP::OpenMP_C)
TARGET_LINK_LIBRARIES(ftle_cuda_pin m OpenMP::OpenMP_C)

INSTALL(TARGETS ftle_cuda ftle_cuda_pin ftle_static ftle_dynamic ftle_guided  RUNTIME DESTINATION bin)
