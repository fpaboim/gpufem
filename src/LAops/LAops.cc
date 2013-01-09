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

////////////////////////////////////////////////////////////////////////////////
// LAops.cc - Basic Linear Algebra Functions
// Author: Francisco Paulo de Aboim (fpaboim@gmail.com)
////////////////////////////////////////////////////////////////////////////////

#include "LAops/LAops.h"

#include <math.h>
#include <time.h>
#include <windows.h>
#include <assert.h>
#include <omp.h>
#include <cstdio>

#include "OpenCL/OCLwrapper.h"
#include "Utils/util.h"
#include "SPRmatrix/SPRmatrix.h"

////////////////////////////////////////////////////////////////////////////////
// Computes dot product of two vectors
////////////////////////////////////////////////////////////////////////////////
fem_float dotProduct(int n, fem_float* v1, fem_float* v2) {
  fem_float sum = 0.0;
  int i;
  for (i = 0; i < n; ++i)
    sum += v1[i] * v2[i];

  return sum;
}
// Same function parallelized with OpenMP
fem_float dotProductOMP(int n, fem_float* v1, fem_float* v2) {
  fem_float sum = 0.0;
#pragma omp parallel for reduction (+: sum)
  for (int i = 0; i < n; ++i)
    sum += v1[i] * v2[i];

  return sum;
}

////////////////////////////////////////////////////////////////////////////////
// Makes input matrix the identity matrix
////////////////////////////////////////////////////////////////////////////////
void GetEye(int dim, fem_float** mat) {
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      mat[i][j] = 0;
    }
  }
  for (int i = 0; i < dim; ++i) {
    mat[i][i] = 1;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Computes the determinant of a 2x2 or 3x3 matrix
////////////////////////////////////////////////////////////////////////////////
fem_float det(int modeldim, fem_float** Matrix) {
  fem_float det = 0;
  if (modeldim == 2) {
    det = Matrix[0][0] * Matrix[1][1] - Matrix[0][1] * Matrix[1][0];
  } else {
    det = Matrix[0][0] * Matrix[1][1] * Matrix[2][2] +
          Matrix[1][0] * Matrix[2][1] * Matrix[0][2] +
          Matrix[2][0] * Matrix[0][1] * Matrix[1][2] -
          Matrix[0][0] * Matrix[2][1] * Matrix[1][2] -
          Matrix[1][0] * Matrix[0][1] * Matrix[2][2] -
          Matrix[2][0] * Matrix[1][1] * Matrix[0][2];
  }

  return det;
}

////////////////////////////////////////////////////////////////////////////////
// GetInverse - Computes the determinant of a 2x2 or 3x3 matrix
////////////////////////////////////////////////////////////////////////////////
void matInverse(int modeldim, fem_float** matrix, fem_float det,
                fem_float** inverse) {
  if (modeldim == 2) {
    inverse[0][0] =   matrix[1][1] / det;
    inverse[0][1] =  -matrix[0][1] / det;
    inverse[1][0] =  -matrix[1][0] / det;
    inverse[1][1] =   matrix[0][0] / det;
  } else {
    inverse[0][0] =  (matrix[2][2]*matrix[1][1]-matrix[2][1]*matrix[1][2])/det;
    inverse[0][1] = -(matrix[2][2]*matrix[0][1]-matrix[2][1]*matrix[0][2])/det;
    inverse[0][2] =  (matrix[1][2]*matrix[0][1]-matrix[1][1]*matrix[0][2])/det;
    inverse[1][0] = -(matrix[2][2]*matrix[1][0]-matrix[2][0]*matrix[1][2])/det;
    inverse[1][1] =  (matrix[2][2]*matrix[0][0]-matrix[2][0]*matrix[0][2])/det;
    inverse[1][2] = -(matrix[1][2]*matrix[0][0]-matrix[1][0]*matrix[0][2])/det;
    inverse[2][0] =  (matrix[2][1]*matrix[1][0]-matrix[2][0]*matrix[1][1])/det;
    inverse[2][1] = -(matrix[2][1]*matrix[0][0]-matrix[2][0]*matrix[0][1])/det;
    inverse[2][2] =  (matrix[1][1]*matrix[0][0]-matrix[1][0]*matrix[0][1])/det;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Multiplies lxm matrix by mxn matrix
////////////////////////////////////////////////////////////////////////////////
void matMult(fem_float** matrixC, int l, int m, int n, fem_float** matrixA,
             fem_float** matrixB) {
  int i, j, k;

  for (i = 0; i < l; ++i) {
    for (j = 0; j < n; ++j) {
      matrixC[i][j]=0;
      for (k = 0; k < m; k++)
        matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Multiplies a matrix by a scalar
////////////////////////////////////////////////////////////////////////////////
void matXScalar(int m, int n, fem_float** matrixA, fem_float scalar) {
  int i, j;

  for (i = 0; i < m; ++i)
    for (j = 0; j < n; ++j)
      matrixA[i][j] = matrixA[i][j] * scalar;
}

////////////////////////////////////////////////////////////////////////////////
// Multiplies a matrix by a scalar
////////////////////////////////////////////////////////////////////////////////
void matPMat(int m, int n, fem_float** matrixA, fem_float** matrixB) {
  int i, j;

  for (i = 0; i < m; ++i)
    for (j = 0; j < n; ++j)
      matrixA[i][j] = matrixA[i][j] + matrixB[i][j];
}

////////////////////////////////////////////////////////////////////////////////
// Creates a transposed mxn matrix
////////////////////////////////////////////////////////////////////////////////
void matTranp(fem_float** transp, int m, int n, fem_float** matrix) {
  int i, j;
  for (i = 0; i < n; ++i)
    for (j = 0; j < m; ++j)
      transp[i][j] = matrix[j][i];
}

////////////////////////////////////////////////////////////////////////////////
// Scalar Alpha X Plus Y -> y = alpha*x + y
////////////////////////////////////////////////////////////////////////////////
void addScaledVectToSelf(fem_float* vector_self, const fem_float* vector_add,
                         const fem_float alpha, int n ) {
  int i;
  for (i = 0; i < n; ++i) {
    vector_self[i] = (alpha * vector_add[i]) + vector_self[i];
  }
}
// Parallel version with OpenMP
void addScaledVectToSelfOMP(fem_float* vector_self, const fem_float* vector_add,
                            const fem_float alpha, int n ) {
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
      vector_self[i] = (alpha * vector_add[i]) + vector_self[i];
    }
}

////////////////////////////////////////////////////////////////////////////////
// Scalar Alpha X Plus Y -> y = x + beta*y
////////////////////////////////////////////////////////////////////////////////
void addSelfScaledToVect(fem_float* vector_self, const fem_float* vector_add,
                         const fem_float beta, int n ) {
  int i;
  for (i = 0; i < n; ++i) {
    vector_self[i] = vector_add[i] + (beta*vector_self[i]);
  }
}
// Parallel version with OpenMP
void addSelfScaledToVectOMP(fem_float* vector_self, const fem_float* vector_add,
                            const fem_float beta, int n ) {
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
      vector_self[i] = vector_add[i] + (beta*vector_self[i]);
    }
}


////////////////////////////////////////////////////////////////////////////////
// Computes a matrix vector product A*x = y (SEMV)
////////////////////////////////////////////////////////////////////////////////
int spax_y(SPRmatrix* matrix_A, const fem_float* vector_X, fem_float* vector_Y,
           int n, bool print ) {
  int i, j;

  for (i = 0; i < n; ++i) {
    vector_Y[i] = 0;
    for (j = 0; j < n; ++j)
      vector_Y[i] +=  matrix_A->GetElem(i,j) * vector_X[j];
  }

  return 1;
}

////////////////////////////////////////////////////////////////////////////////
// Computes a matrix vector product A*x = y (SEMV)
////////////////////////////////////////////////////////////////////////////////
int spax_yOMP(SPRmatrix* matrix_A, const fem_float* vector_X,
              fem_float* vector_Y, int n, bool print ) {
  #pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    vector_Y[i] = 0;
    for (int j = 0; j < n; ++j)
      vector_Y[i] = vector_Y[i] + (matrix_A->GetElem(i,j) * vector_X[j]);
  }

  return 1;
}
