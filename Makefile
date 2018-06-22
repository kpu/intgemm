CXX := g++
CXXFLAGS := -Wall -Werror -fPIC -O3 -march=native -std=c++11
SRC := avx512_gemm.cc avx2_gemm.cc sse2_gemm.cc SSE_Matrix_Mult.cc StopWatch.cc 
OBJ := ${SRC:.cc=.o}

all: Test QuantizeTest Benchmark

avx512_gemm.o: avx512_gemm.h avx512_gemm.cc
	${CXX} ${CXXFLAGS} -c -mavx512bw -mavx512vl avx512_gemm.cc -o avx512_gemm.o

sse2_gemm.o: multiply.h interleave.h sse2_gemm.cc sse2_gemm.h
	${CXX} ${CXXFLAGS} -c sse2_gemm.cc -o sse2_gemm.o

Test: ${OBJ} Test.o
	${CXX} ${CXXFLAGS} ${OBJ} Test.o -o Test

Benchmark: ${OBJ} Benchmark.o
	${CXX} ${CXXFLAGS} ${OBJ} Benchmark.o -o Benchmark

QuantizeTest: QuantizeTest.o StopWatch.o avx512_gemm.o avx2_gemm.o sse2_gemm.o
	${CXX} ${CXXFLAGS} QuantizeTest.o avx512_gemm.o avx2_gemm.o sse2_gemm.o -o QuantizeTest

.c.o: interleave.h multiply.h
	${CXX} ${CXXFLAGS} -c $<

clean:
	rm -f ${OBJ} QuantizeTest Test Test.o QuantizeTest.o Benchmark.o
