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
#include "SPRmatrix/CSRmatrix.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "LAops/LAops.h"

//-----------------------------------------------------------------------
CSRmatrix::CSRmatrix(int matdim) {
  m_matdim      = matdim;
  m_initrowsize = 128;
  m_growstep    = 128;
  AllocateMatrix(matdim);
}

//-----------------------------------------------------------------------
CSRmatrix::~CSRmatrix() {
  Teardown();
}
//-----------------------------------------------------------------------

////////////////////////////////////////////////////////////////////////////////
// Matrix Manipulation Functions
////////////////////////////////////////////////////////////////////////////////

// SetElem: sets sparse matrix element to val
////////////////////////////////////////////////////////////////////////////////
void CSRmatrix::SetElem(const int row, const int col, const fem_float val) {
  if (!InputIsOK(val, row, col))
    return;

  // if diagonal of element found adds value to existing entry
  int pos = BinSearchInt2(m_ROWdata[row].col, col, m_ROWdata[row].nNZ);
  if (m_ROWdata[row].col[pos] == col) {
    m_ROWdata[row].val[pos] = val;
  } else {  // else inserts new element
    // increments number of non-zeros
    m_ROWdata[row].nNZ++;
    // if necessary resizes row array
    if (m_ROWdata[row].nNZ>m_ROWdata[row].maxSz)
      GrowROW(row);
    // Moves Memory: (to , from , blocksize)
    memmove(&m_ROWdata[row].col[pos+1], &m_ROWdata[row].col[pos],
            (m_ROWdata[row].nNZ-(pos+1))*sizeof(int) );
    memmove(&m_ROWdata[row].val[pos+1], &m_ROWdata[row].val[pos],
            (m_ROWdata[row].nNZ-(pos+1))*sizeof(fem_float) );
    m_ROWdata[row].val[pos] = val;
    m_ROWdata[row].col[pos] = col;
  }
}

// AddElem: adds number to sparse matrix
////////////////////////////////////////////////////////////////////////////////
void CSRmatrix::AddElem(const int row, const int col, const fem_float val) {
  if (!InputIsOK(val, row, col))
    return;

  // if diagonal of element found adds value to existing entry
  int pos = BinSearchInt2(m_ROWdata[row].col, col, m_ROWdata[row].nNZ);
  if (m_ROWdata[row].col[pos] == col) {
    m_ROWdata[row].val[pos] += val;
  } else {  // inserts new element
    // increments number of non-zeros
    m_ROWdata[row].nNZ++;
    // if necessary resizes row array
    if (m_ROWdata[row].nNZ >= m_ROWdata[row].maxSz)
      GrowROW(row);
    // Moves Memory: (to, from, blocksize)
    memmove(&m_ROWdata[row].col[pos+1], &m_ROWdata[row].col[pos],
            (m_ROWdata[row].nNZ-(pos))*sizeof(int) );
    memmove(&m_ROWdata[row].val[pos+1], &m_ROWdata[row].val[pos],
            (m_ROWdata[row].nNZ-(pos+1))*sizeof(fem_float) );
    m_ROWdata[row].val[pos] = val;
    m_ROWdata[row].col[pos] = col;
  }
}

// GetElem: gets number from sparse matrix
////////////////////////////////////////////////////////////////////////////////
fem_float CSRmatrix::GetElem(const int row, const int col) {
  if (!BoundsOK(row, col))
    return 0;

  int pos = BinSearchInt2(m_ROWdata[row].col, col, m_ROWdata[row].nNZ);

  if (m_ROWdata[row].col[pos] == col)
    return m_ROWdata[row].val[pos];

  return 0;
}

// GetMatSize: gets matrix size, does not consider size of pointers!
////////////////////////////////////////////////////////////////////////////////
size_t CSRmatrix::GetMatSize() {
  int nelem = 0;
  for (int i = 0; i < m_matdim; i++) {
    nelem += m_ROWdata[i].nNZ;
  }

  return (nelem*(sizeof(fem_float)+sizeof(int)));
}

// GetNNZ: gets number of non-zero entries in matrix
////////////////////////////////////////////////////////////////////////////////
int CSRmatrix::GetNNZ() {
  int nnz = 0;
  for (int i = 0; i < m_matdim; ++i) {
    nnz += m_ROWdata[i].nNZ;
  }
  return nnz;
}

// SetNNZ: Sets the number of non-zero entries in matrix
////////////////////////////////////////////////////////////////////////////////
void CSRmatrix::SetNNZInfo(int nnz, int band) {
  if (nnz < 0 || band < 0)
    return;

}

// Ax_y: does matrix vector multiply and stores result in y (which should be
// allocated by calling function)
////////////////////////////////////////////////////////////////////////////////
void CSRmatrix::Ax_y(fem_float* x, fem_float* y) {
  spax_yOMP(this, x, y, m_matdim, false);
}

// GPU_CG: Solves Ax = b for x by conjugate gradient method using GPU
////////////////////////////////////////////////////////////////////////////////
void CSRmatrix::SolveCgGpu(fem_float* vector_X,
                           fem_float* vector_B,
                           int n_iterations,
                           fem_float epsilon,
                           size_t local_work_size){
}

// Clear: resets the matrix
////////////////////////////////////////////////////////////////////////////////
void CSRmatrix::Clear() {
  // Frees mallocd value and column info
  Teardown();
  m_initrowsize = 128;
  AllocateMatrix(m_matdim);
}

// Teardown: frees memory used by matrix
////////////////////////////////////////////////////////////////////////////////
void CSRmatrix::Teardown() {
  // Frees mallocd value and column info
  if (m_ROWdata) {
    for (int i = 0; i < m_matdim; ++i) {
      free(m_ROWdata[i].val);
      _aligned_free(m_ROWdata[i].col);
    }
    free(m_ROWdata); m_ROWdata = NULL;
  }
}

// GrowROW: reallocates row memory to make room for new NZ entries
////////////////////////////////////////////////////////////////////////////////
inline void CSRmatrix::GrowROW(const int row) {
  m_ROWdata[row].maxSz += m_growstep;
  m_ROWdata[row].val = (fem_float*) realloc(m_ROWdata[row].val,
                                    (m_ROWdata[row].maxSz * sizeof(fem_float)));
  m_ROWdata[row].col = (int*)_aligned_realloc(m_ROWdata[row].col,
                                     (m_ROWdata[row].maxSz * sizeof(int)),
                                     16);
}

// AllocateMatrix: allocated memory for the matrix
////////////////////////////////////////////////////////////////////////////////
int CSRmatrix::AllocateMatrix(const int matdim) {
  m_ROWdata    = (ROWdata*)malloc(matdim*sizeof(ROWdata));
  for (int i = 0; i < matdim; ++i) {
    m_ROWdata[i].val   = (fem_float*)malloc(m_initrowsize*sizeof(fem_float));
    m_ROWdata[i].col   = (int*)_aligned_malloc(m_initrowsize*sizeof(int), 16);
    m_ROWdata[i].col[0]= INT_MAX;
    m_ROWdata[i].nNZ   = 0;
    m_ROWdata[i].maxSz = m_initrowsize;
  }

  return 1;
}
