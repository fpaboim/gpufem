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
// Sparsely Sparse Matrix Library - Compressed Row Sparse Matrix Implementation
// Author: Francisco Aboim
// TecGraf / PUC-RIO
////////////////////////////////////////////////////////////////////////////////

// Headers
#include "SPRmatrix/CSRmatrix2.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "LAops/LAops.h"

//-----------------------------------------------------------------------
CSRmatrix2::CSRmatrix2(int matdim) {
  m_matdim   = matdim;
  m_initsize = 128;
  m_growstep = 128;

  AllocateMatrix(m_initsize);
}

//-----------------------------------------------------------------------
CSRmatrix2::~CSRmatrix2() {
}
//-----------------------------------------------------------------------

////////////////////////////////////////////////////////////////////////////////
// Matrix Manipulation Functions
////////////////////////////////////////////////////////////////////////////////

// SetElem: sets sparse matrix element to val
////////////////////////////////////////////////////////////////////////////////
void CSRmatrix2::SetElem(const int row, const int col, const fem_float val) {
}

// AddElem: adds number to sparse matrix
////////////////////////////////////////////////////////////////////////////////
void CSRmatrix2::AddElem(const int row, const int col, const fem_float val) {
}

// GetElem: gets number from sparse matrix
////////////////////////////////////////////////////////////////////////////////
fem_float CSRmatrix2::GetElem(const int row, const int col) {

  return 0;
}

// GetMatSize: gets matrix size, does not consider size of pointers!
////////////////////////////////////////////////////////////////////////////////
size_t CSRmatrix2::GetMatSize() {
  return 1;
}

// GetNNZ: gets number of non-zero entries in matrix
////////////////////////////////////////////////////////////////////////////////
int CSRmatrix2::GetNNZ() {
  return 1;
}

// SetNNZ: Sets the number of non-zero entries in matrix
////////////////////////////////////////////////////////////////////////////////
void CSRmatrix2::SetNNZInfo(int nnz, int band) {
  AllocateMatrix(nnz);
}

// Ax_y: does matrix vector multiply and stores result in y (which should be
// allocated by calling function)
////////////////////////////////////////////////////////////////////////////////
void CSRmatrix2::Ax_y(fem_float* x, fem_float* y) {
  spax_yOMP(this, x, y, m_matdim, false);
}

// GPU_CG: Solves Ax = b for x by conjugate gradient method using GPU
////////////////////////////////////////////////////////////////////////////////
void CSRmatrix2::SolveCgGpu(fem_float* vector_X,
                           fem_float* vector_B,
                           int n_iterations,
                           fem_float epsilon,
                           size_t local_work_size){
}

// Clear: resets the matrix
////////////////////////////////////////////////////////////////////////////////
void CSRmatrix2::Clear() {
  // Frees mallocd value and column info
  Teardown();
  AllocateMatrix(m_matdim);
}

// Teardown: frees memory used by matrix
////////////////////////////////////////////////////////////////////////////////
void CSRmatrix2::Teardown() {
}

// GrowROW: reallocates row memory to make room for new NZ entries
////////////////////////////////////////////////////////////////////////////////
inline void CSRmatrix2::GrowROW(const int row) {

}

// AllocateMatrix: allocated memory for the matrix
////////////////////////////////////////////////////////////////////////////////
int CSRmatrix2::AllocateMatrix(const int allocsize) {
  m_matdata = (fem_float*)malloc(allocsize * sizeof(fem_float));

  return 1;
}


