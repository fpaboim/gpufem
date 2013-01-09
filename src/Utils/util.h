/////////////////////////////////////////////////////////////////////
// util.cpp Header File
/////////////////////////////////////////////////////////////////////
#ifndef util_H
#define util_H

#include <vector>

#define FEM_ERR 0
#define FEM_OK  1
#define fem_float float

// Console Output Utility Functions
void        printMatrix  (fem_float** matrix, int m, int n);
void        printMatrixRM(fem_float*  matrix, int m, int n);
void        printVectorf (fem_float*  vec, int n);
void        printVectori (int*  vec, int n);
void        printMatrixSTL(std::vector<std::vector<int>> matrix);

// Memory Allocation Utility Functions
void        zeroMatrix (fem_float** matrix, int m, int n);
void        zeroMatrixV(fem_float*  matrix, int m, int n);
fem_float*  allocVector(int n, bool initAs0);
fem_float** allocMatrix(int m, int n, bool initAs0);
void        freeInnerVectorsF (fem_float** matrix, int m);
void        freeInnerVectorsI (int** matrix, int m);


#endif