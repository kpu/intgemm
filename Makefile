CXX := g++
CXXFLAGS := -Wall -Werror -fPIC -O3 -march=native
SRC := AVX_Matrix_Mult.cc  SSE_Matrix_Mult.cc  StopWatch.cc  Test.cc
OBJ := ${SRC:.cc=.o}

all: Test

Test: ${OBJ}
	${CXX} ${CXXFLAGS} ${OBJ} -o Test

.c.o:
	${CXX} ${CXXFLAGS} -c $<
