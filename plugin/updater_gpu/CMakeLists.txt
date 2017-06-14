cmake_minimum_required (VERSION 3.2)
project (updater_gpu)

set(GPU_COMPUTE_VER 50;52;60;61 CACHE STRING
    "Space separated list of compute versions to be built against")
find_package(CUDA REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3 -funroll-loops -msse2 -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES")
endif()


# Make sure we are using C++11
# Visual Studio 12.0 and newer supports enough c++11 to make this work
if(MSVC)
  if(MSVC_VERSION LESS 1800)
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
  endif()
else()
  # GCC 4.6 with c++0x supports enough to make this work
  include(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
  CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)

  set(STD_FLAG "")
  if(COMPILER_SUPPORTS_CXX11)
    set(STD_FLAG "-std=c++11")
  elseif(COMPILER_SUPPORTS_CXX0X)
    set(STD_FLAG "-std=c++0x")
  else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${STD_FLAG}")
endif()

#Make sure we are using the static runtime
if(MSVC)
    set(variables
        CMAKE_C_FLAGS_DEBUG
        CMAKE_C_FLAGS_MINSIZEREL
        CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_RELWITHDEBINFO
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_MINSIZEREL
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_RELWITHDEBINFO
    )
    foreach(variable ${variables})
        if(${variable} MATCHES "/MD")
            string(REGEX REPLACE "/MD" "/MT" ${variable} "${${variable}}")
        endif()
    endforeach()
endif()

file(GLOB SOURCES
  src/*.cu
  src/exact/*.cu
)
include_directories(src)
include_directories(src/exact)
include(${PROJECT_SOURCE_DIR}/cmake/Utils.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/Cuda.cmake)

#Set library output directories
if(MSVC)
    #With MSVC shared library is considered runtime
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/lib)
else()
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
    #Prevent shared library being called liblibxgboost.so on Linux
    set(CMAKE_SHARED_LIBRARY_PREFIX "")
endif()

#Find googletest
set(GTEST_DIRECTORY "${CACHE_PREFIX}" CACHE PATH "Googletest directory")
include_directories(${GTEST_DIRECTORY}/include)

set(JON 1)

if(JON)
  #gencode flags
  include_directories(${CMAKE_SOURCE_DIR}/nccl/src)

  set(GPU_COMPUTE_VER 50;52;60;61 CACHE STRING
    "Space separated list of compute versions to be built against")

  set(GENCODE_FLAGS "")
  foreach(ver ${GPU_COMPUTE_VER})
    set(GENCODE_FLAGS "${GENCODE_FLAGS}-gencode arch=compute_${ver},code=sm_${ver};")
  endforeach()
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--expt-extended-lambda;${GENCODE_FLAGS};-lineinfo;")
  if(NOT MSVC)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xcompiler -fPIC")
  endif()
  
  # use below for forcing specific arch
  #cuda_compile(CUDA_OBJS ${SOURCES} ${CUDA_NVCC_FLAGS})
  cuda_compile(CUDA_OBJS ${SOURCES} ${CUDA_NVCC_FLAGS})
  set(LINK_LIBRARIES ${LINK_LIBRARIES} ${CUDA_LIBRARIES})
  
  #  cuda_add_library(${PROJECT_NAME} OBJECT ${CUDA_OBJS})
  add_library(${PROJECT_NAME} OBJECT ${SOURCES})
  set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
  
else()

  set(GPU_COMPUTE_VER 50;52;60;61 CACHE STRING
    "Space separated list of compute versions to be built against")
  
  #  gencode flags
  set(GENCODE_FLAGS "")
  foreach(ver ${GPU_COMPUTE_VER})
    set(GENCODE_FLAGS "${GENCODE_FLAGS}-gencode arch=compute_${ver},code=sm_${ver};")
  endforeach()
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--expt-extended-lambda;${GENCODE_FLAGS};-lineinfo;")
  if(NOT MSVC)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xcompiler -fPIC")
  endif()
  
  #use below for forcing specific arch
  cuda_compile(CUDA_OBJS ${CUDA_SOURCES} ${CUDA_NVCC_FLAGS})  
  set(LINK_LIBRARIES ${LINK_LIBRARIES} ${CUDA_LIBRARIES})
  
  cuda_add_library(${PROJECT_NAME} OBJECT ${CUDA_OBJS})
  set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
endif()
