cmake_minimum_required(VERSION 3.1)
project(intgemm)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#Only the benchmark uses C++11
set(CMAKE_CXX_STANDARD 11)

set_source_files_properties(sse2_gemm.cc PROPERTIES COMPILE_FLAGS -msse2)
set_source_files_properties(ssse3_gemm.cc PROPERTIES COMPILE_FLAGS "-msse2 -mssse3")
set_source_files_properties(avx2_gemm.cc PROPERTIES COMPILE_FLAGS -mavx2)
set_source_files_properties(avx512_gemm.cc PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512bw -mavx512dq")

try_compile(AVX512 "${CMAKE_BINARY_DIR}/compile_tests" "${CMAKE_SOURCE_DIR}/compile_test_avx512.cc"
  #Hack: pass compiler arguments as definitions because the test code overrides CXX_FLAGS :'(
  COMPILE_DEFINITIONS -mavx512f -mavx512bw -mavx512dq)

set(GEMMS
  avx2_gemm.cc
  sse2_gemm.cc
  ssse3_gemm.cc)

if (AVX512)
  set(GEMMS avx512_gemm.cc ${GEMMS})
else()
  message("Not building AVX512-based multiplication because your compiler is too old.\nFor details rerun cmake with --debug-trycompile then try to build in compile_tests/CMakeFiles/CMakeTmp.")
  set_source_files_properties(intgemm.cc quantize_test.cc multiply_test.cc benchmark.cc PROPERTIES COMPILE_DEFINITIONS "INTGEMM_NO_AVX512")
endif()

add_library(intgemm STATIC ${GEMMS} intgemm.cc)
foreach(exe example benchmark)
  add_executable(${exe} ${exe}.cc)
  target_link_libraries(${exe} intgemm)
endforeach()

include_directories(.)
add_executable(tests test/multiply_test.cc test/quantize_test.cc)
target_link_libraries(tests intgemm)
