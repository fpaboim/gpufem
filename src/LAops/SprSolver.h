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
#ifndef SPRSOLVER_H
#define SPRSOLVER_H

#include "LAops/LAops.h"
#include "Utils/util.h"
#include "SPRmatrix/SPRmatrix.h"


// Direct Solvers
//****************************************************************************//
// Solves by cholesky decomposition
int cholesky(SPRmatrix* matrixA,
             float* vectorX,
             float* vectorY,
             int dim,
             bool print);

// Iterative Solvers
//****************************************************************************//
// CPU Conjugate Gradient Solver
int  CPU_CG(SPRmatrix* matrix_A,
            fem_float* vector_X,
            fem_float* vector_Y,
            int n,
            int n_iterations,
            fem_float epsilon,
            bool print );

// GPU Conjugate Gradient Solver
void GPU_CG(const fem_float** matrix_A,
            fem_float* vector_X,
            fem_float* vector_B,
            int n,
            int n_iterations,
            fem_float epsilon,
            size_t local_work_size);

#endif
