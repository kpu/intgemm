cmake_minimum_required(VERSION 3.5)
project(intgemm)
string(ASCII 27 Esc)
set(Orange "${Esc}[33m")
set(ColourReset "${Esc}[m")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 11)

try_compile(AVX512 "${CMAKE_BINARY_DIR}/compile_tests" "${CMAKE_SOURCE_DIR}/compile_test_avx512.cc"
  #Hack: pass compiler arguments as definitions because the test code overrides CXX_FLAGS :'(
  COMPILE_DEFINITIONS -mavx512f -mavx512bw -mavx512dq)

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  # Working around https://bugs.llvm.org/show_bug.cgi?id=41482
  # Anything compiled with clang might not work properly in SSE2/SSSE3 world
  message("${Orange}Compiling with Clang and using -mavx2 due to https://bugs.llvm.org/show_bug.cgi?id=41482. Support for SSE2/SSSE3 is likely broken.${ColourReset}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
endif()

if(NOT AVX512)
  message("${Orange}Not building AVX512-based multiplication because your compiler is too old.\nFor details rerun cmake with --debug-trycompile then try to build in compile_tests/CMakeFiles/CMakeTmp.${ColourReset}")
  set_source_files_properties(example.cc test/quantize_test.cc test/multiply_test.cc benchmark.cc intgemm.cc PROPERTIES COMPILE_DEFINITIONS "INTGEMM_NO_AVX512")
endif()

foreach(exe benchmark biasmultiply)
  add_executable(${exe} benchmarks/${exe}.cc intgemm.cc)
endforeach()

add_executable(example example.cc intgemm.cc)

include_directories(.)
add_executable(tests
  test/common.cc
  test/multiply_test.cc
  test/quantize_test.cc
  test/relu_test.cc
  test/add127_test.cc
  intgemm.cc
)

#CTest integration with Catch2
include(CMake/Catch.cmake)
include(CTest)
catch_discover_tests(tests)
