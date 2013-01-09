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
#include "DENmatrix.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "LAops/LAops.h"
#include "LAops/SprSolver.h"

//-----------------------------------------------------------------------
DENmatrix::DENmatrix(int matdim) {
  m_matdim = matdim;
  m_data   = (fem_float**)malloc(m_matdim * sizeof(fem_float*));
  for (int i = 0; i < m_matdim; i++) {
    m_data[i] = (fem_float*)calloc(m_matdim , sizeof(fem_float));
  }
}

//-----------------------------------------------------------------------
DENmatrix::~DENmatrix() {
  Teardown();
}
//-----------------------------------------------------------------------

////////////////////////////////////////////////////////////////////////////////
// Matrix Manipulation Functions
////////////////////////////////////////////////////////////////////////////////

// SetElem: sets sparse matrix element to val
////////////////////////////////////////////////////////////////////////////////
void DENmatrix::SetElem(const int row, const int col, const fem_float val) {
  if (!InputIsOK(val, row, col))
    return;

  m_data[row][col] = val;
}

// AddElem: adds number to sparse matrix
////////////////////////////////////////////////////////////////////////////////
void DENmatrix::AddElem(const int row, const int col, const fem_float val) {
  if (!InputIsOK(val, row, col))
    return;

  m_data[row][col] += val;
}

// GetElem: gets number from sparse matrix
////////////////////////////////////////////////////////////////////////////////
fem_float DENmatrix::GetElem(const int row, const int col) {
  if (!BoundsOK(row, col))
    return 0;

  return m_data[row][col];
}

// GetMatSize: gets matrix size, does not consider size of pointers!
////////////////////////////////////////////////////////////////////////////////
size_t DENmatrix::GetMatSize() {
  return (m_matdim * m_matdim * sizeof(fem_float));
}


// GetNNZ: gets number of non-zero entries in matrix
////////////////////////////////////////////////////////////////////////////////
int DENmatrix::GetNNZ() {
  int nnz=0;
  for (int i=0; i<m_matdim; ++i) {
    for (int j=0; j<m_matdim; j++) {
      if(m_data[i][j] != 0)
        nnz++;
    }
  }
  return nnz;
}

// SetNNZ: Sets the number of non-zero entries in matrix
////////////////////////////////////////////////////////////////////////////////
void DENmatrix::SetNNZInfo(int nnz, int band) {
}

// Ax_y: does matrix vector multiply and stores result in y (which should be
// allocated by calling function)
////////////////////////////////////////////////////////////////////////////////
void DENmatrix::Ax_y(fem_float* x, fem_float* y) {
  spax_yOMP(this, x, y, m_matdim, false);
}

// GPU_CG: Solves Ax = b for x by conjugate gradient method using GPU
////////////////////////////////////////////////////////////////////////////////
void DENmatrix::SolveCgGpu(fem_float* vector_X,
                           fem_float* vector_B,
                           int n_iterations,
                           fem_float epsilon,
                           size_t local_work_size){
}

// Clear: Clears matrix by tearing it down and reallocating memory
////////////////////////////////////////////////////////////////////////////////
void DENmatrix::Clear() {
  Teardown();
  m_data    = (fem_float**)malloc(m_matdim*sizeof(fem_float*));
  for(int i=0; i<m_matdim; i++)
    m_data[i] = (fem_float*)calloc(m_matdim,sizeof(fem_float));
}

// Teardown: frees memory used by matrix
////////////////////////////////////////////////////////////////////////////////
void DENmatrix::Teardown() {
  if (m_data != NULL) {
    // Frees dyanmically alloced matrix
    for (int i = 0; i < m_matdim; ++i)
      free(m_data[i]);
    free(m_data);
    m_data = NULL;
  }
}
