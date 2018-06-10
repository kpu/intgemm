CXX := g++
CXXFLAGS := -DNDEBUG -Wall -Werror -fPIC -O3 -march=native -mavx512vl -mavx512bw
SRC := AVX_Matrix_Mult.cc  SSE_Matrix_Mult.cc Quantize.cc StopWatch.cc
OBJ := ${SRC:.cc=.o}

all: Test QuantizeTest Benchmark

Test: ${OBJ} Test.o
	${CXX} ${CXXFLAGS} ${OBJ} Test.o -o Test

Benchmark: ${OBJ} Benchmark.o
	${CXX} ${CXXFLAGS} ${OBJ} Benchmark.o -o Benchmark

QuantizeTest: Quantize.o QuantizeTest.o StopWatch.o
	${CXX} ${CXXFLAGS} Quantize.o QuantizeTest.o StopWatch.o -o QuantizeTest

.c.o: AVX_Matrix_Mult.h
	${CXX} ${CXXFLAGS} -c $<

clean:
	rm -f ${OBJ} QuantizeTest Test
