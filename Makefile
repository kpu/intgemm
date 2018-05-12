CXX := g++
CXXFLAGS := -Wall -Werror -fPIC -O3 -march=native -mavx512vl
SRC := AVX_Matrix_Mult.cc  SSE_Matrix_Mult.cc  StopWatch.cc
OBJ := ${SRC:.cc=.o}

all: Test QuantizeTest

Test: ${OBJ} Test.o
	${CXX} ${CXXFLAGS} ${OBJ} Test.o -o Test

QuantizeTest: ${OBJ} QuantizeTest.o
	${CXX} ${CXXFLAGS} ${OBJ} QuantizeTest.o -o QuantizeTest

.c.o:
	${CXX} ${CXXFLAGS} -c $<
