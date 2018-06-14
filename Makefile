CXX := g++
CXXFLAGS := -DNDEBUG -Wall -Werror -fPIC -O3 -march=native
SRC := avx512_gemm.cc avx2_gemm.cc SSE_Matrix_Mult.cc Quantize.cc StopWatch.cc
OBJ := ${SRC:.cc=.o}

all: Test QuantizeTest Benchmark

avx512_gemm.o: avx512_gemm.h avx512_gemm.cc
	${CXX} ${CXXFLAGS} -c -mavx512bw -mavx512vl avx512_gemm.cc -o avx512_gemm.o

Test: ${OBJ} Test.o
	${CXX} ${CXXFLAGS} ${OBJ} Test.o -o Test

Benchmark: ${OBJ} Benchmark.o
	${CXX} ${CXXFLAGS} ${OBJ} Benchmark.o -o Benchmark

QuantizeTest: Quantize.o QuantizeTest.o StopWatch.o avx512_gemm.o avx2_gemm.o
	${CXX} ${CXXFLAGS} Quantize.o QuantizeTest.o StopWatch.o avx512_gemm.o avx2_gemm.o -o QuantizeTest

.c.o: AVX_Matrix_Mult.h
	${CXX} ${CXXFLAGS} -c $<

clean:
	rm -f ${OBJ} QuantizeTest Test
