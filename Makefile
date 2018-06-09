CXX := g++
CXXFLAGS := -Wall -Werror -fPIC -O3 -march=native -mavx512vl -mavx512bw
SRC := AVX_Matrix_Mult.cc  SSE_Matrix_Mult.cc Quantize.cc StopWatch.cc
OBJ := ${SRC:.cc=.o}

all: Test QuantizeTest

Test: ${OBJ} Test.o
	${CXX} ${CXXFLAGS} ${OBJ} Test.o -o Test

QuantizeTest: Quantize.o QuantizeTest.o StopWatch.o
	${CXX} ${CXXFLAGS} Quantize.o QuantizeTest.o StopWatch.o -o QuantizeTest

.c.o: AVX_Matrix_Mult.h
	${CXX} ${CXXFLAGS} -c $<

clean:
	rm -f ${OBJ} QuantizeTest Test
