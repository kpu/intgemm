CXX := g++
CXXFLAGS := -Wall -Werror -fPIC -O3 -march=native
SRC := avx512_gemm.cc avx2_gemm.cc sse2_gemm.cc stop_watch.cc intgemm.cc
OBJ := ${SRC:.cc=.o}

all: test quantize_test benchmark example

avx512_gemm.o: multiply.h interleave.h avx512_gemm.h avx512_gemm.cc
	${CXX} ${CXXFLAGS} -c -mavx512bw -mavx512vl avx512_gemm.cc -o avx512_gemm.o

avx2_gemm.o: multiply.h interleave.h avx2_gemm.h avx2_gemm.cc
	${CXX} ${CXXFLAGS} -c -mavx2 avx2_gemm.cc -o avx2_gemm.o

sse2_gemm.o: multiply.h interleave.h sse2_gemm.cc sse2_gemm.h
	${CXX} ${CXXFLAGS} -c -msse2 -mssse3 sse2_gemm.cc -o sse2_gemm.o

test: ${OBJ} test.o aligned.h
	${CXX} ${CXXFLAGS} ${OBJ} test.o -o test

benchmark: ${OBJ} benchmark.o aligned.h
	${CXX} ${CXXFLAGS} ${OBJ} benchmark.o -o benchmark

quantize_test: ${OBJ} quantize_test.o aligned.h
	${CXX} ${CXXFLAGS} ${OBJ} quantize_test.o -o quantize_test

example: ${OBJ} example.o aligned.h
	${CXX} ${CXXFLAGS} ${OBJ} example.o -o example

.c.o: interleave.h multiply.h
	${CXX} ${CXXFLAGS} -c $<

clean:
	rm -f ${OBJ} quantize_test test benchmark quantize_test.o test.o benchmark.o
