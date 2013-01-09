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

/////////////////////////////////////////////////////////////////////
// LAops.cpp Header File
/////////////////////////////////////////////////////////////////////
#ifndef LAops_H
#define LAops_H

#include "Utils/util.h"
#include "SPRmatrix/SPRmatrix.h"

// Basic Linear Algebra Functions
fem_float dotProduct(int n, fem_float* v1, fem_float* v2);
fem_float dotProductOMP(int n, fem_float* v1, fem_float* v2);
void      GetEye(int dim, fem_float** mat);
fem_float det(int modeldim, fem_float** Matrix);
void      matInverse(int modeldim, fem_float** matrix, fem_float det,
                     fem_float** inverse);
void      matMult(fem_float** matrixC, int l, int m, int n,
                  fem_float** matrixA, fem_float** matrixB);
void      matXScalar(int m, int n, fem_float** matrixA, fem_float scalar);
void      matPMat(int m, int n, fem_float** matrixA, fem_float** matrixB);
void      matTranp(fem_float** transp, int m, int n, fem_float** matrix);
void      addScaledVectToSelf(fem_float* vector_self,
                              const fem_float* vector_add,
                              const fem_float alpha, int n );
void      addScaledVectToSelfOMP(fem_float* vector_self,
                                 const fem_float* vector_add,
                                 const fem_float alpha, int n );
void      addSelfScaledToVect(fem_float* vector_self,
                              const fem_float* vector_add,
                              const fem_float beta, int n );
void      addSelfScaledToVectOMP(fem_float* vector_self,
                                 const fem_float* vector_add,
                                 const fem_float beta, int n );
int       spax_y(SPRmatrix* matrix_A, const fem_float* vector_X,
                 fem_float* vector_Y, int n, bool print);
int       spax_yOMP(SPRmatrix* matrix_A, const fem_float* vector_X,
                    fem_float* vector_Y, int n, bool print);

#endif
