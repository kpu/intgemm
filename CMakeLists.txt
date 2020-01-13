cmake_minimum_required(VERSION 3.5)
project(intgemm)
string(ASCII 27 Esc)
set(Orange "${Esc}[33m")
set(ColourReset "${Esc}[m")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 11)

# Check if compiler supports AVX512
try_compile(INTGEMM_COMPILER_SUPPORTS_AVX512
  ${CMAKE_CURRENT_BINARY_DIR}/compile_tests
  ${CMAKE_CURRENT_SOURCE_DIR}/compile_test_avx512.cc
  #Hack: pass compiler arguments as definitions because the test code overrides CXX_FLAGS :'(
  COMPILE_DEFINITIONS -mavx512f -mavx512bw -mavx512dq)

if(NOT INTGEMM_COMPILER_SUPPORTS_AVX512)
  message("${Orange}Not building AVX512-based multiplication because your compiler is too old.\nFor details rerun cmake with --debug-trycompile then try to build in compile_tests/CMakeFiles/CMakeTmp.${ColourReset}")
endif()

try_compile(INTGEMM_COMPILER_SUPPORTS_AVX512VNNI
  ${CMAKE_CURRENT_BINARY_DIR}/compile_tests
  ${CMAKE_CURRENT_SOURCE_DIR}/compile_test_avx512vnni.cc)
#No compiler flags for this test; that's part of the test!
if(NOT INTGEMM_COMPILER_SUPPORTS_AVX512VNNI)
  message("${Orange}Not building AVX512VNNI-based multiplication because your compiler is too old.\nFor details rerun cmake with --debug-trycompile then try to build in compile_tests/CMakeFiles/CMakeTmp.${ColourReset}")
endif()

# Working around https://bugs.llvm.org/show_bug.cgi?id=41482
# Anything compiled with clang might not work properly in SSE2/SSSE3 world
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  message("${Orange}Compiling with Clang and using -mavx2 due to https://bugs.llvm.org/show_bug.cgi?id=41482. Support for SSE2/SSSE3 is likely broken.${ColourReset}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
endif()

# Generate configure file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/intgemm_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/intgemm_config.h)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

foreach(exe benchmark biasmultiply)
  add_executable(${exe} benchmarks/${exe}.cc intgemm.cc)
endforeach()

add_executable(example example.cc intgemm.cc)

add_executable(tests
  test/test.cc

  # General tests
  test/add127_test.cc
  test/multiply_test.cc
  test/quantize_test.cc
  test/rearrangement_b_test.cc
  test/utils_test.cc

  # Kernels tests
  test/kernels/add_bias_test.cc
  test/kernels/bitwise_not_test.cc
  test/kernels/downcast_test.cc
  test/kernels/exp_test.cc
  test/kernels/floor_test.cc
  test/kernels/multiply_sat_test.cc
  test/kernels/multiply_test.cc
  test/kernels/quantize_test.cc
  test/kernels/relu_test.cc
  test/kernels/rescale_test.cc
  test/kernels/sigmoid_test.cc
  test/kernels/tanh_test.cc
  test/kernels/unquantize_test.cc
  test/kernels/upcast_test.cc
  test/kernels/write_test.cc

  # Definitions
  intgemm.cc
)

#CTest integration with Catch2
include(${CMAKE_CURRENT_SOURCE_DIR}/CMake/Catch.cmake)
include(CTest)
catch_discover_tests(tests)
