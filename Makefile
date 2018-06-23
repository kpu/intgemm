CXX := g++
CXXFLAGS := -Wall -Werror -fPIC -O3 -march=native -DNDEBUG
SRC := avx512_gemm.cc avx2_gemm.cc sse2_gemm.cc stop_watch.cc dispatch.cc
OBJ := ${SRC:.cc=.o}

all: test quantize_test benchmark

avx512_gemm.o: avx512_gemm.h avx512_gemm.cc
	${CXX} ${CXXFLAGS} -c -mavx512bw -mavx512vl avx512_gemm.cc -o avx512_gemm.o

sse2_gemm.o: multiply.h interleave.h sse2_gemm.cc sse2_gemm.h
	${CXX} ${CXXFLAGS} -c sse2_gemm.cc -o sse2_gemm.o

test: ${OBJ} test.o
	${CXX} ${CXXFLAGS} ${OBJ} test.o -o test

benchmark: ${OBJ} benchmark.o
	${CXX} ${CXXFLAGS} ${OBJ} benchmark.o -o benchmark

quantize_test: ${OBJ} quantize_test.o
	${CXX} ${CXXFLAGS} ${OBJ} quantize_test.o -o quantize_test

.c.o: interleave.h multiply.h
	${CXX} ${CXXFLAGS} -c $<

clean:
	rm -f ${OBJ} quantize_test test benchmark quantize_test.o test.o benchmark.o
