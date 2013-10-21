// Copyright 2012, Francisco Aboim (fpaboim@tecgraf.puc-rio.br)
// All Rights Reserved
//
// This Program is licensed under the MIT License as follows:
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <iostream>
#include <fstream>
#include <vector>
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>

#include "Utils/util.h"

using namespace std;


/******************************************************************************/
//                 Console Output Utility Functions                           //
/******************************************************************************/

////////////////////////////////////////////////////////////////////////////////
// Prints out a Matrix to Console
////////////////////////////////////////////////////////////////////////////////
void printMatrix(fem_float** matrix, int m, int n) {
  int i, j;

  printf("[ ");
  for(i=0; i<m; i++)
  {
    for(j=0; j<n; j++)
      printf("%2.1f ", matrix[i][j]);
    printf("%s",(i != (m-1)) ? "\n  " : "]\n");
  }
}

////////////////////////////////////////////////////////////////////////////////
// Prints out a Matrix to Console
////////////////////////////////////////////////////////////////////////////////
void printMatrixRM(fem_float* matrix, int m, int n) {
  int i, j;

  printf("[ ");
  for(i=0; i<m; i++)
  {
    for(j=0; j<n; j++)
      printf("%4.3f ", matrix[m*i+j]);
    printf("%s",(i != (m-1)) ? "\n  " : "]\n");
  }
}

////////////////////////////////////////////////////////////////////////////////
// Prints out a Vector to Console
////////////////////////////////////////////////////////////////////////////////
void printVectorf(fem_float* vec, int n) {
  int i;

  printf("[");
  for(i=0; i<n; i++)
  {
    printf(" %8.4f ", vec[i]);
    printf("%s",(i != (n-1)) ? "\n " : "]\n");
  }
}

////////////////////////////////////////////////////////////////////////////////
// Prints out a Vector to Console
////////////////////////////////////////////////////////////////////////////////
void printVectori(int* vec, int n) {
  int i;

  printf("[");
  for(i=0; i<n; i++)
  {
    printf(" %i ", vec[i]);
    printf("%s",(i != (n-1)) ? "\n " : "]\n");
  }
}


////////////////////////////////////////////////////////////////////////////////
// Prints out a Vector to Console
////////////////////////////////////////////////////////////////////////////////
void printMatrixSTL(vector<vector<int>> matrix) {

  printf("[ ");
  for(size_t i=0; i<matrix.size(); i++)
  {
    size_t rowlen = matrix[i].size();
    for(size_t j=0; j<rowlen; j++)
      printf("%i ", matrix[i][j]);
    printf("\n  ");
  }
}


/******************************************************************************/
//                 Memory Allocation Utility Functions                        //
/******************************************************************************/

////////////////////////////////////////////////////////////////////////////////
// Makes all Elements Inside a Matrix 0
////////////////////////////////////////////////////////////////////////////////
void zeroMatrix( fem_float** matrix, int m, int n )
{
  int i, j;

#pragma omp parallel for private(i,j)
  for( i=0; i<m; i++)
    for( j=0; j<n; j++)
      matrix[i][j] = 0;
}

////////////////////////////////////////////////////////////////////////////////
// Makes all Elements Inside a Matrix(row major vector format) 0
////////////////////////////////////////////////////////////////////////////////
void zeroMatrixV( fem_float* matrix, int m, int n )
{
  int i, j;

#pragma omp parallel for private(i,j)
  for( i=0; i<m; i++)
    for( j=0; j<n; j++)
      matrix[(i*n)+j] = 0;
}


////////////////////////////////////////////////////////////////////////////////
// Dynamically allocates a vector
////////////////////////////////////////////////////////////////////////////////
fem_float* allocVector(int n, bool initAs0) {
  fem_float* vector  = (fem_float*)malloc(n*sizeof(fem_float));

  if (initAs0 == true) {
#pragma omp parallel for
    for(int i=0; i<n; i++)
        vector[i] = 0;
  }

  return vector;
}

////////////////////////////////////////////////////////////////////////////////
// Dynamically allocates a matrix - all pointers must be deallocated by
// user using FreeInnerVectors and then freeing matrix outer (**) pointer
////////////////////////////////////////////////////////////////////////////////
fem_float** allocMatrix( int m, int n, bool initAs0 ) {
  int i, j;
  fem_float** matrix;

  matrix = (fem_float**)malloc(m*sizeof(fem_float*));
  for( i=0; i<m; i++)
    matrix[i] = (fem_float*)malloc(n*sizeof(fem_float));

  if ( initAs0 == true )
  {
#pragma omp parallel for private(i,j)
    for( i=0; i<m; i++)
      for( j=0; j<n; j++)
        matrix[i][j] = 0;
  }

  return matrix;
}

////////////////////////////////////////////////////////////////////////////////
fem_float** copyMatrixF(int m, int n, fem_float** inputmatrix) {
  if (inputmatrix == NULL)
    return NULL;
  for (int i = 0; i < m; i++) {
    if (inputmatrix[i] == NULL) {
      return NULL;
    }
  }
  fem_float** outmatrix;
  outmatrix = (fem_float**)malloc(m*sizeof(fem_float*));
  for(int i=0; i<m; i++) {
    outmatrix[i] = (fem_float*)malloc(n*sizeof(fem_float));
  }
  for(int i=0; i<m; i++)
    memcpy(outmatrix[i], inputmatrix[i], n*sizeof(fem_float));

  return outmatrix;
}

////////////////////////////////////////////////////////////////////////////////
int** copyMatrixI(int m, int n, int** inputmatrix) {
  if (inputmatrix == NULL)
    return NULL;
  for (int i = 0; i < m; i++) {
    if (inputmatrix[i] == NULL) {
      return NULL;
    }
  }
  int** outmatrix;
  outmatrix = (int**)malloc(m*sizeof(int*));
  for(int i=0; i<m; i++) {
    outmatrix[i] = (int*)malloc(n*sizeof(int));
  }
  for(int i=0; i<m; i++)
    memcpy(outmatrix[i], inputmatrix[i], n*sizeof(int));

  return outmatrix;
}

////////////////////////////////////////////////////////////////////////////////
fem_float* copyVectorF(int size, fem_float* inputvec) {
  if (inputvec == NULL)
    return NULL;
  fem_float* outvec;
  size_t memsize = size * sizeof(fem_float);
  outvec = (fem_float*)malloc(memsize);
  memcpy(outvec, inputvec, memsize);

  return outvec;
}

////////////////////////////////////////////////////////////////////////////////
int* copyVectorI(int size, int* inputvec) {
  if (inputvec == NULL)
    return NULL;
  int* outvec;
  size_t memsize = size * sizeof(int);
  outvec = (int*)malloc(memsize);
  memcpy(outvec, inputvec, memsize);

  return outvec;
}

////////////////////////////////////////////////////////////////////////////////
// Frees dynamically allocated matrices
////////////////////////////////////////////////////////////////////////////////
void freeInnerVectorsF(fem_float** matrix, int m) {
  for (int i = 0; i < m; i++) {
    if (matrix[i]) {
      free(matrix[i]);
      matrix[i] = NULL;
    }
  }
  if (matrix) {
    free(matrix);
    matrix = NULL;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Frees dynamically allocated matrices
////////////////////////////////////////////////////////////////////////////////
void freeInnerVectorsI(int** matrix, int m) {
  for (int i = 0; i < m; i++) {
    if (matrix[i]) {
      free(matrix[i]);
      matrix[i] = NULL;
    }
  }
  if (matrix) {
    free(matrix);
    matrix = NULL;
  }
}
