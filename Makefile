CXX := g++
CXXFLAGS := -Wall -Werror -fPIC -O3 -msse4.2
INC := 
LIBS := 

all: SSE_Matrix_Mult Vector_Functions

SSE_Matrix_Mult: SSE_Matrix_Mult.cpp
	$(CXX) $(CXXFLAGS) $(INC) -o SSE_Matrix_Mult SSE_Matrix_Mult.cpp $(LIBS)

Vector_Functions: Vector_Functions.cpp
	$(CXX) $(CXXFLAGS) $(INC) -o Vector_Functions Vector_Functions.cpp $(LIBS)

clean:
	rm -f SSE_Matrix_Mult Vector_Functions
