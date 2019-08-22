#include "intgemm.cc"
#include "aligned.h"
#include <iostream>
#include <random>
#include <string>
#include <algorithm>
#include <fstream>
#include <sstream>


/*Adapted from https://www.bfilipek.com/2018/07/string-view-perf-followup.html . We should probably go string_view way
inline void tokenizeLine(std::string& str, std::vector<std::string>& output,
 std::string delimeter = " ") {
    auto first = std::begin(str);

    while (first != str.end()) {
        const auto second = std::find_first_of(first, std::end(str), std::begin(delimeter), std::end(delimeter));

        if (first != second) {
            output.emplace_back(str.substr(std::distance(std::begin(str), first), std::distance(first, second)));
        }

        if (second == str.end())
            break;

        first = std::next(second);
    }
}

//This is a different parsing method, without stringStream
template<class StringType>
void ReadInFile(StringType infile) {
	std::ifstream in(infile);
	std::string line;

	//First line, Info about the matrix
	std::getline(in, line);
    std::istringstream iss(line);
    std::string temp1, temp2, temp3, temp4;
    int RowsA, ColsA, RowsB, ColsB;
    if (!(iss >> temp1 >> RowsA >> temp2 >> ColsA >> temp3 >> RowsB >> temp4 >> ColsB)) {
    	std::cerr << "Error parsing line 1 " << std::endl;
    	exit(1);
    }

    //Second line, get QuantMult
    std::getline(in, line);
    std::istringstream iss2(line);
    float quantMultA, quantMultB;
    if (!(iss2 >> temp1 >> quantMultA >> temp2 >> quantMultA)) { 
    	std::cerr << "Error parsing line 2 " << std::endl;
    	exit(1);
    }
    std::getline(in, line); //Just some text
    //Fourth line, AQuant
    std::vector<int> AQuant;
    std::getline(in, line);
    std::vector<std::string> tmp_container;
    tokenizeLine(line, tmp_container);
    if (tmp_container.size() != RowsA*ColsA) {
    	std::cerr << "Error parsing matrix A. Size mismatch. Expected " <<  RowsA*ColsA << " got " << tmp_container.size() << std::endl;
    }
    for (auto&& num : tmp_container) {
    	AQuant.push_back(std::stoi(num));
    }
    tmp_container.resize(0);

    std::getline(in, line); //Just some text
    //Sixth line, B_raw
    std::vector<float> B_raw;
    std::getline(in, line);
    tokenizeLine(line, tmp_container);
    if (tmp_container.size() != RowsB*ColsB) {
    	std::cerr << "Error parsing matrix B. Size mismatch. Expected " <<  RowsB*ColsB << " got " << tmp_container.size() << std::endl;
    }
    for (auto&& num : tmp_container) {
    	B_raw.push_back(std::stof(num));
    }
    tmp_container.resize(0);

    std::getline(in, line); //Just some text
    //Eight line, Bias
    std::vector<float> Bias;
    std::getline(in, line);
    tokenizeLine(line, tmp_container);
    if (tmp_container.size() != ColsB) {
    	std::cerr << "Error parsing bias. Size mismatch. Expected " <<  ColsB << " got " << tmp_container.size() << std::endl;
    }
    for (auto&& num : tmp_container) {
    	Bias.push_back(std::stof(num));
    }
    tmp_container.resize(0);

}

*/
template<class StringType>
void ReadInFile(StringType infile) {
	std::ifstream in(infile);
	std::string line;

	//First line, Info about the matrix
	std::getline(in, line);
    std::istringstream iss(line);
    std::string temp1, temp2, temp3, temp4;
    int RowsA, ColsA, RowsB, ColsB;
    if (!(iss >> temp1 >> RowsA >> temp2 >> ColsA >> temp3 >> RowsB >> temp4 >> ColsB)) {
    	std::cerr << "Error parsing line 1 " << std::endl;
    	exit(1);
    }

    //Second line, get QuantMult
    std::getline(in, line);
    std::istringstream iss2(line);
    float quantMultA, quantMultB;
    if (!(iss2 >> temp1 >> quantMultA >> temp2 >> quantMultA)) { 
    	std::cerr << "Error parsing line 2 " << std::endl;
    	exit(1);
    }
    std::getline(in, line); //Just some text for human readability

    //4th line, AQuant
    std::vector<int> AQuant;
    std::getline(in, line);
    std::istringstream iss3(line);
    for (int i = 0; i < RowsA*ColsA; i++) {
    	int num;
    	if (!(iss3 >> num)) {
    		std::cerr << "Error parsing matrix A at " << i << std::endl;;
    	}
    	AQuant.push_back(num);
    }

    std::getline(in, line); //Just some text for human readability
    //6th line, B_raw
    std::vector<float> B_raw;
    std::getline(in, line);
    std::istringstream iss4(line);
    for (int i = 0; i < RowsB*ColsB; i++) {
    	float num;
    	if (!(iss4 >> num)) {
    		std::cerr << "Error parsing matrix B " << std::endl;
    	}
    	B_raw.push_back(num);
    }

    std::getline(in, line); //Just some text for human readability
    //8th line, Bias
    std::vector<float> Bias;
    std::getline(in, line);
    std::istringstream iss5(line);
    for (int i = 0; i < ColsB; i++) {
    	float num;
    	if (!(iss5 >> num)) {
    		std::cerr << "Error parsing matrix bias " << std::endl;
    	}
    	Bias.push_back(num);
    }
}

using namespace intgemm;
template<class T>
void printMatrix(T* data, Index rows, Index cols) {
	std::cout << "[";
	for (int i = 0; i<rows; i++) {
		std::cout << "[";
		for (int j =0; j<cols; j++) {
			std::cout << (float)data[i*cols + j];
			if (j != cols - 1) {
				std::cout << ", ";
			}
		}
		std::cout << "]";
		if (i != rows -1) {
			std::cout << ',' << std::endl;
		}
	}
	std::cout << "]" << std::endl;
}

void SlowRefFloat(const float *A, const float *B, float *C, Index A_rows, Index width, Index B_cols, const float *bias) {
  for (Index r = 0; r < A_rows; ++r) {
    for (Index c = 0; c < B_cols; ++c) {
      float sum = 0.0f;
      for (Index k = 0; k < width; ++k) {
        sum += A[r * width + k] * B[k * B_cols + c];
      }
      if (bias) {
        C[r * B_cols + c] = sum + bias[c];
      } else {
        C[r * B_cols + c] = sum;
      }
    }
  }
}

// Compute A*B slowly from integers.
template <class Integer> 
void SlowRefInt(const Integer *A, const Integer *B, float *C, float unquant_mult, Index A_rows, Index width, Index B_cols, const float *bias) {
  for (Index r = 0; r < A_rows; ++r) {
    for (Index c = 0; c < B_cols; ++c) {
      int32_t sum = 0;
      for (Index k = 0; k < width; ++k) {
        sum += static_cast<int16_t>(A[r * width + k]) * static_cast<int16_t>(B[k * B_cols + c]);
      }
      if (bias) {
        C[r * B_cols + c] = sum * unquant_mult + bias[c];
      } else {
        C[r * B_cols + c] = sum * unquant_mult;
      }
    }
  }
}

int main() {

	const Index A_rows = 1;
	const Index width = 2048;
	const Index B_cols = 8;

	AlignedVector<float> A(A_rows * width);
    AlignedVector<float> B(width * B_cols);
    AlignedVector<float> bias(B_cols);

    float alpha = 2.0f;
    float quant_mult = 127/alpha;
    float unquant_mult = 1.0 / (quant_mult * quant_mult);

	std::mt19937 gen;
	std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
	
	for (auto& it : A) {
		it = dist(gen);
	}
	for (auto& it : B) {
		it = dist(gen);
	}
	for (auto& it : bias) {
		it = dist(gen);
	}

	AlignedVector<float> bias_orig(B_cols);
	for (int i = 0; i < bias.size(); i++) {
		bias_orig[i] = bias[i];
	}

    AlignedVector<int8_t> A_prep(A.size());
    AlignedVector<int8_t> B_prep(B.size());

    AVX2_8bit::PrepareA(A.begin(), A_prep.begin(), quant_mult, A_rows, width);
    AVX2_8bit::PrepareB(B.begin(), B_prep.begin(), quant_mult, width, B_cols);
/*
    std::cout << "A:" << std::endl;
    printMatrix(A.begin(), A_rows, width);
    std::cout << "B:" << std::endl;
    printMatrix(B.begin(), width, B_cols);
    std::cout << "bias:" << std::endl;
    printMatrix(bias.begin(), 1, B_cols);*/


    AlignedVector<float> test_C(A_rows * B_cols);

    AVX2_8bit::Multiply(A_prep.begin(), B_prep.begin(), BiasAddUnquantizeC(test_C.begin(), bias.begin(), unquant_mult), A_rows, width, B_cols);
    //AVX2_8bit::Multiply(A_prep.begin(), B_prep.begin(), JustUnquantizeC(test_C.begin(), unquant_mult), A_rows, width, B_cols);
    std::cout << "Old multiply:" << std::endl;
    printMatrix(test_C.begin(), A_rows, B_cols);

    //NEEEXT
    AlignedVector<uint8_t> A_prep2(A.size());
    AVX2_8bit::PrepareA(A.begin(), A_prep2.begin(), quant_mult, A_rows, width);

    AVX2_8bit::PrepareBiasFor8(B.begin(), bias.begin(), alpha, width, B_cols);

    //printMatrix(bias.begin(), 1, B_cols); //Print bias

    AVX2_8bit::Multiply8new(reinterpret_cast<uint8_t*>(A_prep2.begin()), B_prep.begin(), BiasAddUnquantizeC(test_C.begin(), bias.begin(), unquant_mult), A_rows, width, B_cols);
    //AVX2_8bit::Multiply8new(reinterpret_cast<uint8_t*>(A_prep.begin()), B_prep.begin(), JustUnquantizeC(test_C.begin(), unquant_mult), A_rows, width, B_cols);
    
    AlignedVector<int16_t> A_prep3(A.size());
    AlignedVector<int16_t> B_prep3(B.size());
    std::cout << "New multiply:" << std::endl;
    printMatrix(test_C.begin(), A_rows, B_cols);
    for (int i = 0; i < A_prep2.size(); i++) {
        A_prep3[i] = A_prep2[i];
    }
    AVX2_16bit::PrepareB(B.begin(), B_prep3.begin(), quant_mult, width, B_cols);
    AVX2_16bit::Multiply(A_prep3.begin(), B_prep3.begin(), BiasAddUnquantizeC(test_C.begin(), bias.begin(), unquant_mult), A_rows, width, B_cols);
    
    std::cout << "New multiply, 16 bit:" << std::endl;
    printMatrix(test_C.begin(), A_rows, B_cols);

    //FULL INTS
    AlignedVector<float> C_slowint(A_rows * B_cols);
    AlignedVector<int8_t> B_quant(width * B_cols);
    AVX2_8bit::Quantize(B.begin(), B_quant.begin(), quant_mult, B.size());

    SlowRefInt(A_prep.begin(), B_quant.begin(), C_slowint.begin(),
     unquant_mult, A_rows, width, B_cols, bias_orig.begin());


    std::cout << "Reference int8:" << std::endl;
    printMatrix(C_slowint.begin(), A_rows, B_cols);

    //FULL INT16
    AlignedVector<int16_t> A_prep4(A.size());
    for (int i = 0; i < A_prep2.size(); i++) {
        A_prep4[i] = A_prep[i];
    }

    AlignedVector<float> C_slowint2(A_rows * B_cols);
    AlignedVector<int16_t> B_quant2(width * B_cols);
    AVX2_16bit::Quantize(B.begin(), B_quant2.begin(), quant_mult, B.size());

    SlowRefInt(A_prep4.begin(), B_quant2.begin(), C_slowint2.begin(),
     unquant_mult, A_rows, width, B_cols, bias_orig.begin());


    std::cout << "Reference int16:" << std::endl;
    printMatrix(C_slowint2.begin(), A_rows, B_cols);

    //FLOATS
    AlignedVector<float> C(A_rows * B_cols);

	SlowRefFloat(A.begin(), B.begin(), C.begin(), A_rows, width, B_cols, bias_orig.begin());
	std::cout << "Reference float:" << std::endl;
	printMatrix(C.begin(), A_rows, B_cols);

}
